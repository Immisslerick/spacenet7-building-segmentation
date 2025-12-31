from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window
import imageio.v2 as imageio

# ---- must match your training model ----
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

def to_rgb(img_c):
    # img_c: (C,H,W) normalized; make a displayable 0-255 RGB
    c, h, w = img_c.shape
    if c >= 3:
        r, g, b = img_c[2], img_c[1], img_c[0]
        rgb = np.stack([r, g, b], axis=-1)
    else:
        rgb = np.repeat(img_c[0][..., None], 3, axis=-1)

    rgb = rgb.astype(np.float32)
    rgb = rgb - rgb.min()
    if rgb.max() > 0:
        rgb = rgb / rgb.max()
    return (rgb * 255).clip(0, 255).astype(np.uint8)

def main():
    project_root = Path(__file__).resolve().parent.parent
    split_csv = project_root / "runs" / "sn7_split.csv"
    masks_dir = project_root / "data" / "processed" / "masks_png"
    weights = project_root / "runs" / "baseline_unet.pt"

    out_dir = project_root / "runs" / "pred_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

    device = "cpu"
    # infer channel count from first val tif
    with rasterio.open(val_df.loc[0, "tif_path"]) as src:
        in_ch = src.count

    model = TinyUNet(in_ch=in_ch, base=32).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    rng = random.Random(42)
    n_examples = 12
    crop = 512
    thr = 0.35

    for i in range(1, n_examples + 1):
        row = val_df.iloc[rng.randrange(len(val_df))]
        stem = row["filename"]
        tif_path = row["tif_path"]
        mask_path = masks_dir / f"{stem}_mask.png"

        mask_full = imageio.imread(mask_path)
        mask_full = (mask_full > 127).astype(np.uint8)

        with rasterio.open(tif_path) as src:
            H, W = src.height, src.width
            ch = min(crop, H)
            cw = min(crop, W)
            y0 = 0 if H <= ch else rng.randrange(0, H - ch)
            x0 = 0 if W <= cw else rng.randrange(0, W - cw)
            window = Window(x0, y0, cw, ch)
            img = src.read(window=window).astype(np.float32)

        gt = mask_full[y0:y0+ch, x0:x0+cw]

        # normalize same as training
        mean = img.mean(axis=(1, 2), keepdims=True)
        std = img.std(axis=(1, 2), keepdims=True) + 1e-6
        img_n = (img - mean) / std

        x = torch.from_numpy(img_n).unsqueeze(0)  # (1,C,H,W)
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].numpy()

        pred = (prob > thr).astype(np.uint8)

        rgb8 = to_rgb(img_n)
        overlay_gt = rgb8.copy()
        overlay_pred = rgb8.copy()
        overlay_gt[gt == 1] = np.array([255, 0, 0], dtype=np.uint8)
        overlay_pred[pred == 1] = np.array([0, 255, 0], dtype=np.uint8)

        imageio.imwrite(out_dir / f"{i:02d}_rgb.png", rgb8)
        imageio.imwrite(out_dir / f"{i:02d}_gt.png", (gt * 255).astype(np.uint8))
        imageio.imwrite(out_dir / f"{i:02d}_pred.png", (pred * 255).astype(np.uint8))
        imageio.imwrite(out_dir / f"{i:02d}_overlay_gt.png", overlay_gt)
        imageio.imwrite(out_dir / f"{i:02d}_overlay_pred.png", overlay_pred)

    print("Wrote gallery to:", out_dir)

if __name__ == "__main__":
    main()
