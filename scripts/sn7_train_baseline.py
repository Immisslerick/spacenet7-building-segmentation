from pathlib import Path
import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import rasterio
from rasterio.windows import Window
import imageio.v2 as imageio


# ----------------------------
# Dataset: random window crops from tif + cached mask png
#   - Windowed tif read (fast)
#   - Worker-aware RNG (avoids duplicate crops across workers)
# ----------------------------
class SN7RandomCropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, masks_dir: Path, crop: int = 512, samples_per_epoch: int = 2500, base_seed: int = 42):
        self.df = df.reset_index(drop=True)
        self.masks_dir = masks_dir
        self.crop = crop
        self.samples_per_epoch = samples_per_epoch
        self.base_seed = base_seed

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Worker-aware deterministic RNG: stable + non-overlapping across workers
        wi = torch.utils.data.get_worker_info()
        wid = 0 if wi is None else wi.id
        rng = np.random.default_rng(self.base_seed + idx + wid * 1_000_000)

        # Pick a random image row
        row = self.df.iloc[int(rng.integers(0, len(self.df)))]
        stem = row["filename"]
        tif_path = row["tif_path"]
        mask_path = self.masks_dir / f"{stem}_mask.png"

        # Load mask once (H,W), then crop
        mask_full = imageio.imread(mask_path)
        mask_full = (mask_full > 127).astype(np.float32)

        crop = self.crop

        # Open tif, pick random crop coords, then read ONLY that window
        with rasterio.open(tif_path) as src:
            H, W = src.height, src.width

            # Clamp crop size if ever needed (just in case)
            ch = min(crop, H)
            cw = min(crop, W)

            y0 = 0 if H <= ch else int(rng.integers(0, H - ch))
            x0 = 0 if W <= cw else int(rng.integers(0, W - cw))

            window = Window(x0, y0, cw, ch)
            img_c = src.read(window=window).astype(np.float32)  # (C, ch, cw)

        m_c = mask_full[y0:y0+ch, x0:x0+cw]

        # Per-channel normalize (simple baseline)
        mean = img_c.mean(axis=(1, 2), keepdims=True)
        std = img_c.std(axis=(1, 2), keepdims=True) + 1e-6
        img_c = (img_c - mean) / std

        # Light aug (flip)
        if rng.random() < 0.5:
            img_c = img_c[:, :, ::-1].copy()
            m_c = m_c[:, ::-1].copy()
        if rng.random() < 0.5:
            img_c = img_c[:, ::-1, :].copy()
            m_c = m_c[::-1, :].copy()

        x = torch.from_numpy(img_c)                    # (C, H, W)
        y = torch.from_numpy(m_c).unsqueeze(0)         # (1, H, W)

        return x, y


# ----------------------------
# Tiny UNet-ish model
# ----------------------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class TinyUNet(nn.Module):
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


# ----------------------------
# Loss + metric
# ----------------------------
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()

@torch.no_grad()
def iou_score(logits, targets, thr=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()
    inter = (pred * targets).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


def main():
    project_root = Path(__file__).resolve().parent.parent
    split_csv = project_root / "runs" / "sn7_split.csv"
    masks_dir = project_root / "data" / "processed" / "masks_png"

    df = pd.read_csv(split_csv)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    # "Use as many cores as it wants" — DataLoader workers (processes)
    # On i9-9900K: os.cpu_count() likely 16 (threads). 6–8 is usually a sweet spot.
    num_workers = min(8, os.cpu_count() or 0)
    print("DataLoader num_workers:", num_workers, flush=True)

    # Training knobs
    crop = 512
    batch_size = 4
    epochs = 6
    train_samples_per_epoch = 2500
    val_samples_per_epoch = 400

    train_ds = SN7RandomCropDataset(train_df, masks_dir, crop=crop, samples_per_epoch=train_samples_per_epoch, base_seed=42)
    val_ds = SN7RandomCropDataset(val_df, masks_dir, crop=crop, samples_per_epoch=val_samples_per_epoch, base_seed=123)

    pin = (device == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Infer channel count from one sample
    x0, _ = train_ds[0]
    in_ch = x0.shape[0]
    print("Input channels:", in_ch, flush=True)

    model = TinyUNet(in_ch=in_ch, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t_epoch0 = time.perf_counter()

        # Batch progress + ETA
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = 0.7 * bce(logits, y) + 0.3 * dice_loss(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item()

            if step % 25 == 0:
                elapsed = time.perf_counter() - t_epoch0
                sec_per_batch = elapsed / step
                est_remaining_sec = (len(train_loader) - step) * sec_per_batch
                print(
                    f"  ep{ep} batch {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} "
                    f"sec/batch={sec_per_batch:.3f} "
                    f"ETA(min)={est_remaining_sec/60:.1f}",
                    flush=True
                )

        # Validation
        model.eval()
        val_iou = 0.0
        n = 0
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            val_iou += iou_score(logits, y)
            n += 1
        val_iou /= max(1, n)

        epoch_sec = time.perf_counter() - t_epoch0
        print(
            f"Epoch {ep}/{epochs}  "
            f"train_loss={total_loss/len(train_loader):.4f}  "
            f"val_IoU={val_iou:.4f}  "
            f"epoch_min={epoch_sec/60:.1f}",
            flush=True
        )

    out = project_root / "runs" / "baseline_unet.pt"
    torch.save(model.state_dict(), out)
    print("Saved:", out, flush=True)


if __name__ == "__main__":
    main()
