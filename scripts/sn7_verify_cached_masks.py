from pathlib import Path
import random
import numpy as np
import pandas as pd
import rasterio
import imageio.v2 as imageio

def main():
    project_root = Path(__file__).resolve().parent.parent
    split_path = project_root / "runs" / "sn7_subset.csv"   # uses the same subset you cached
    masks_dir  = project_root / "data" / "processed" / "masks_png"
    out_dir    = project_root / "runs" / "verify_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_path)
    random.seed(42)
    picks = df.sample(n=min(6, len(df)), random_state=42)

    for i, row in enumerate(picks.itertuples(index=False), start=1):
        stem = row.filename
        tif_path = Path(row.tif_path)
        mask_path = masks_dir / f"{stem}_mask.png"

        if not mask_path.exists():
            print("Missing mask:", mask_path)
            continue

        with rasterio.open(tif_path) as src:
            img = src.read()  # (C,H,W)

        mask = imageio.imread(mask_path)
        mask = (mask > 127).astype(np.uint8)  # 0/1

        # make quick RGB preview
        if img.shape[0] >= 3:
            r,g,b = img[2], img[1], img[0]
            rgb = np.stack([r,g,b], axis=-1).astype(np.float32)
            rgb = rgb - rgb.min()
            if rgb.max() > 0:
                rgb = rgb / rgb.max()
            rgb8 = (rgb * 255).clip(0,255).astype(np.uint8)
        else:
            band = img[0].astype(np.float32)
            band = band - band.min()
            if band.max() > 0:
                band = band / band.max()
            rgb8 = (band * 255).astype(np.uint8)

        overlay = rgb8.copy()
        if overlay.ndim == 3:
            overlay[mask == 1] = np.array([255, 0, 0], dtype=np.uint8)

        imageio.imwrite(out_dir / f"{i:02d}_image.png", rgb8)
        imageio.imwrite(out_dir / f"{i:02d}_mask.png", (mask * 255).astype(np.uint8))
        imageio.imwrite(out_dir / f"{i:02d}_overlay.png", overlay)

    print("Wrote overlays to:", out_dir)

if __name__ == "__main__":
    main()
