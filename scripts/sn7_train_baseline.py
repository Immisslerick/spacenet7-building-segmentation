from pathlib import Path
import os
import time
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from multiprocessing import Value

os.environ.setdefault("GDAL_CACHEMAX", "4096")  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Spend RAM to reduce TIFF block re-reads (MB)


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
        self._mask_cache = OrderedDict()
        self._src_cache = OrderedDict()
        self.max_masks = 512     # tune up (RAM is plentiful)
        self.max_open_tifs = 8   # keep this modest to avoid "too many open files"

        self.epoch = Value("i", 0)


    def set_epoch(self, epoch: int):
        self.epoch.value = int(epoch)


    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
    # Worker-aware deterministic RNG: stable + non-overlapping across workers
        ep = int(self.epoch.value)
        wi = torch.utils.data.get_worker_info()
        wid = 0 if wi is None else wi.id
        seed = self.base_seed + ep * 10_000_000 + idx + wid * 1_000_000
        rng = np.random.default_rng(seed)

        # ---- knobs (only these three) ----
        min_pos = 64         # minimum positive pixels in mask crop to accept
        max_tries = 10       # how many times to try to find a positive crop
        keep_empty_p = 0.4  # probability to accept an empty crop anyway
        crop = self.crop
        # ----------------------------------

        # We will pick a row + crop; prefer crops with buildings (mask positives)
        for attempt in range(max_tries):
            # Pick a random image row
            row = self.df.iloc[int(rng.integers(0, len(self.df)))]
            stem = row["filename"]
            tif_path = row["tif_path"]
            mask_path = self.masks_dir / f"{stem}_mask.png"

            # ---- load full mask (cached) ----
            mask_full = self._mask_cache.get(mask_path)
            if mask_full is None:
                mask_full = imageio.imread(mask_path)
                # binarize to 0/1 float so pos sums are meaningful
                mask_full = (mask_full > 127).astype(np.float32)
                self._mask_cache[mask_path] = mask_full
                if len(self._mask_cache) > self.max_masks:
                    self._mask_cache.popitem(last=False)
            else:
                self._mask_cache.move_to_end(mask_path)

            # ---- open tif (cached) ----
            src = self._src_cache.get(tif_path)
            if src is None:
                src = rasterio.open(tif_path)
                self._src_cache[tif_path] = src
                if len(self._src_cache) > self.max_open_tifs:
                    _, old = self._src_cache.popitem(last=False)
                    try:
                        old.close()
                    except Exception:
                        pass
            else:
                self._src_cache.move_to_end(tif_path)

            H, W = src.height, src.width
            ch = min(crop, H)
            cw = min(crop, W)

            y0 = 0 if H <= ch else int(rng.integers(0, H - ch + 1))
            x0 = 0 if W <= cw else int(rng.integers(0, W - cw + 1))

            # Crop mask first (cheap)
            m_c = mask_full[y0:y0+ch, x0:x0+cw]
            pos = float(m_c.sum())

            # Accept crop if it has positives, or sometimes accept empty, or last try
            if pos >= min_pos or rng.random() < keep_empty_p or attempt == max_tries - 1:
                window = Window(x0, y0, cw, ch)
                img_c = src.read(window=window).astype(np.float32)  # (C, ch, cw)
                break

        # Pad to (C, crop, crop) if needed
        if img_c.shape[1] != crop or img_c.shape[2] != crop:
            out = np.zeros((img_c.shape[0], crop, crop), dtype=np.float32)
            out[:, :img_c.shape[1], :img_c.shape[2]] = img_c
            img_c = out

            m_out = np.zeros((crop, crop), dtype=np.float32)
            m_out[:m_c.shape[0], :m_c.shape[1]] = m_c
            m_c = m_out

        # Per-channel normalize (simple baseline)
        mean = img_c.mean(axis=(1, 2), keepdims=True)
        std = img_c.std(axis=(1, 2), keepdims=True) + 1e-6
        img_c = (img_c - mean) / std

        # Light aug (flips)
        if rng.random() < 0.5:
            img_c = img_c[:, :, ::-1].copy()
            m_c = m_c[:, ::-1].copy()
        if rng.random() < 0.5:
            img_c = img_c[:, ::-1, :].copy()
            m_c = m_c[::-1, :].copy()
        # Random 90° rotations (satellite-friendly)
        k = int(rng.integers(0, 4))  # 0,1,2,3
        if k:
            img_c = np.rot90(img_c, k, axes=(1, 2)).copy()  # img_c is (C,H,W)
            m_c   = np.rot90(m_c,   k, axes=(0, 1)).copy()  # m_c is (H,W)

        x = torch.from_numpy(img_c)                      # float32
        y = torch.from_numpy(m_c).unsqueeze(0).float()   # (1,H,W) float32 0/1
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

def tp_fp_fn_scores(logits, targets, thrs=(0.25, 0.30, 0.35, 0.40, 0.50)):
    """
    Returns per-threshold TP/FP/FN counts (micro totals over batch+pixels).
    Shapes:
      logits:  (B,1,H,W)
      targets: (B,1,H,W) float {0,1}
    """
    probs = torch.sigmoid(logits)
    tgt = (targets > 0.5)

    tp, fp, fn = [], [], []
    for thr in thrs:
        pred = (probs > thr)
        tp.append((pred & tgt).sum())
        fp.append((pred & ~tgt).sum())
        fn.append((~pred & tgt).sum())

    # float tensors, shape (len(thrs),)
    return torch.stack(tp).float(), torch.stack(fp).float(), torch.stack(fn).float()

def iou_scores(logits, targets, thrs=(0.25, 0.30, 0.35, 0.40, 0.50), eps=1e-6):
    probs = torch.sigmoid(logits)
    outs = []
    for thr in thrs:
        pred = (probs > thr).float()
        inter = (pred * targets).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
        outs.append(((inter + eps) / (union + eps)).mean())
    return torch.stack(outs)  # (len(thrs),)


def worker_init_fn(worker_id: int):
    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    return


def main():
    project_root = Path(__file__).resolve().parent.parent
    split_csv = project_root / "runs" / "sn7_split_subset.csv"
    masks_dir = project_root / "data" / "processed" / "masks_png"

    df = pd.read_csv(split_csv)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
     
    print("Device:", device)
    if device == "cpu":
        torch.set_num_threads(os.cpu_count())       # match physical cores
        torch.set_num_interop_threads(28) 
    print("MAIN torch threads:", torch.get_num_threads(),
        "interop:", torch.get_num_interop_threads(), flush=True)
    print("MAIN OMP:", os.getenv("OMP_NUM_THREADS"),
        "MKL:", os.getenv("MKL_NUM_THREADS"), flush=True)
    
    num_workers = 16
    print("DataLoader num_workers:", num_workers, flush=True)

    # Training knobs
    crop = 512
    batch_size = 4
    epochs = 12
    train_samples_per_epoch = 2000
    val_samples_per_epoch = 200

    train_ds = SN7RandomCropDataset(train_df, masks_dir, crop=crop, samples_per_epoch=train_samples_per_epoch, base_seed=42)
    val_ds = SN7RandomCropDataset(val_df, masks_dir, crop=crop, samples_per_epoch=val_samples_per_epoch, base_seed=123)

    pin = (device == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,

    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        
    )

    # Infer channel count from one sample
    # Infer channel count from one sample
    x0, y0 = train_ds[0]  # <-- change _ to y0

    print("SANITY x:", x0.shape, x0.dtype,
        "min/max:", float(x0.min()), float(x0.max()),
        "mean/std:", float(x0.mean()), float(x0.std()), flush=True)

    print("SANITY y:", y0.shape, y0.dtype,
        "min/max:", float(y0.min()), float(y0.max()),
        "pos_frac:", float(y0.mean()),          # fraction of pixels that are 1's
        "pos_px:", float(y0.sum()), flush=True) # count of positive pixels

    in_ch = x0.shape[0]
    print("Input channels:", in_ch, flush=True)

    model = TinyUNet(in_ch=in_ch, base=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    pos_w = torch.tensor([2.0], device=device, dtype=torch.float32)   
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_val_iou = -1.0
    best_ep = 0
    best_path = project_root / "runs" / "baseline_unet_best.pt"
    last_path = project_root / "runs" / "baseline_unet_last.pt"

    for ep in range(1, epochs + 1):
        train_ds.set_epoch(ep)
        # NOTE: do NOT set val_ds epoch if you want validation to stay fixed

        model.train()
        total_loss = 0.0
        t_epoch0 = time.perf_counter()

        # Batch progress + ETA
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss(logits, y)

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
        thrs = (0.25, 0.30, 0.35, 0.40, 0.50)

        val_loss_sum = 0.0
        eps = 1e-6
        tp_sum = torch.zeros(len(thrs))
        fp_sum = torch.zeros(len(thrs))
        fn_sum = torch.zeros(len(thrs))

        val_iou_sum = torch.zeros(len(thrs))
        n = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                tp, fp, fn = tp_fp_fn_scores(logits, y, thrs=thrs)
                tp_sum += tp.cpu()
                fp_sum += fp.cpu()
                fn_sum += fn.cpu()


                vloss = 0.7 * bce(logits, y) + 0.3 * dice_loss(logits, y)
                val_loss_sum += float(vloss.item())

                val_iou_sum += iou_scores(logits, y, thrs=thrs).cpu()
                n += 1
            prec = (tp_sum + eps) / (tp_sum + fp_sum + eps)
            rec  = (tp_sum + eps) / (tp_sum + fn_sum + eps)
            f1   = (2 * prec * rec + eps) / (prec + rec + eps)


        val_loss = val_loss_sum / max(1, n)
        val_ious = val_iou_sum / max(1, n)

        iou_str = " ".join([f"@{t:.2f}:{float(v):.4f}" for t, v in zip(thrs, val_ious)])
        pr_str = " ".join([f"@{t:.2f} P:{float(p):.4f} R:{float(r):.4f} F1:{float(ff):.4f}"
                   for t, p, r, ff in zip(thrs, prec, rec, f1)])

        print(f"val_loss={val_loss:.4f}  val_IoU {iou_str}  val_PR {pr_str}", flush=True)


        # pick which threshold you want to summarize as "val_IoU="
        val_iou = float(val_ious[thrs.index(0.40)])  # or 0.50 if you prefer

        # Save best checkpoint
        if val_iou > best_val_iou + 1e-6:
            best_val_iou = val_iou
            best_ep = ep
            torch.save(model.state_dict(), last_path)
            print("Saved last:", last_path, flush=True)
            print(f"Best was ep={best_ep} val_IoU={best_val_iou:.4f} -> {best_path}", flush=True)



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
