from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window
import imageio.v2 as imageio


# --- Must match the model used in training ---
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
    def __init__(self, in_ch=4, base=64):
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


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def main():
    project_root = Path(__file__).resolve().parent.parent
    split_csv = project_root / "runs" / "sn7_split.csv"
    masks_dir = project_root / "data" / "processed" / "masks_png"
    weights = project_root / "runs" / "baseline_unet.pt"

    if not split_csv.exists():
        raise FileNotFoundError(f"Missing: {split_csv}")
    if not weights.exists():
        raise FileNotFoundError(f"Missing: {weights} (did training run and save?)")

    df = pd.read_csv(split_csv)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

    # Infer input channels from the first tif
    with rasterio.open(val_df.loc[0, "tif_path"]) as src:
        in_ch = src.count

    device = "cpu"

    sd = torch.load(weights, map_location="cpu")
    # handle either raw state_dict or {"model": state_dict}
    state = sd if "enc1.0.weight" in sd else sd["model"]

    base = int(state["enc1.0.weight"].shape[0])   # 32/48/64/etc
    in_ch_ckpt = int(state["enc1.0.weight"].shape[1])

    print(f"Checkpoint expects in_ch={in_ch_ckpt}, base={base}", flush=True)

    # sanity: your tif channel count should match checkpoint
    if in_ch != in_ch_ckpt:
        raise RuntimeError(f"in_ch mismatch: tif has {in_ch}, checkpoint expects {in_ch_ckpt}")

    model = TinyUNet(in_ch=in_ch, base=base).to(device)
    model.load_state_dict(state)
    model.eval()


    # Sweep settings
    crop = 512
    n_crops = 300  # increase for more stable estimate (slower)
    thresholds = np.arange(0.15, 0.66, 0.05)

    rng = np.random.default_rng(42)
    scores = {float(t): [] for t in thresholds}

    for _ in range(n_crops):
        row = val_df.iloc[int(rng.integers(0, len(val_df)))]
        stem = row["filename"]
        tif_path = row["tif_path"]
        mask_path = masks_dir / f"{stem}_mask.png"

        gt_full = (imageio.imread(mask_path) > 127).astype(np.uint8)

        with rasterio.open(tif_path) as src:
            H, W = src.height, src.width
            ch = min(crop, H)
            cw = min(crop, W)
            y0 = 0 if H <= ch else int(rng.integers(0, H - ch))
            x0 = 0 if W <= cw else int(rng.integers(0, W - cw))
            window = Window(x0, y0, cw, ch)
            img = src.read(window=window).astype(np.float32)

        gt = gt_full[y0:y0+ch, x0:x0+cw]

        # Normalize like training
        mean = img.mean(axis=(1, 2), keepdims=True)
        std = img.std(axis=(1, 2), keepdims=True) + 1e-6
        img_n = (img - mean) / std

        x = torch.from_numpy(img_n).unsqueeze(0)  # (1,C,H,W)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].numpy()

        for t in thresholds:
            pred = (prob > float(t)).astype(np.uint8)
            scores[float(t)].append(iou(pred, gt))

    print("Threshold sweep (mean IoU on random val crops):")
    best_t, best_iou = None, -1.0
    for t in sorted(scores.keys()):
        miou = float(np.mean(scores[t]))
        print(f"  thr={t:.2f}  mean_IoU={miou:.4f}")
        if miou > best_iou:
            best_iou = miou
            best_t = t

    print(f"\nBest threshold: {best_t:.2f}  mean_IoU={best_iou:.4f}")


if __name__ == "__main__":
    main()
