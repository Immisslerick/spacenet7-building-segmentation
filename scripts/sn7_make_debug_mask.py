from pathlib import Path
import numpy as np
import pandas as pd

import rasterio
from rasterio.features import rasterize

from shapely import wkt
from shapely.geometry import mapping
from affine import Affine

def locate_tif(data_root: Path, filename: str) -> Path:
    # Try exact filename first (fast)
    for folder in ["SN7_buildings_train_sample", "SN7_buildings_train"]:
        base = data_root / folder
        if base.exists():
            p = next(base.rglob(filename), None)
            if p:
                return p

    # If the CSV filename omits extension or differs slightly, try stem match
    stem = Path(filename).stem
    for folder in ["SN7_buildings_train_sample", "SN7_buildings_train"]:
        base = data_root / folder
        if base.exists():
            p = next(base.rglob(f"{stem}*.tif"), None)
            if p:
                return p

    raise FileNotFoundError(f"Could not locate tif for CSV filename: {filename}")

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data" / "sn7"

    csv_path = data_root / "SN7_buildings_train_csvs" / "csvs" / "sn7_train_ground_truth_pix.csv"
    df = pd.read_csv(csv_path)

    # Pick a filename that DEFINITELY has labels (highest polygon count)
    fname = df["filename"].value_counts().index[0]
    n_polys = int(df["filename"].value_counts().iloc[0])
    print("Chosen filename:", fname)
    print("Polygon count:", n_polys)

    tif_path = locate_tif(data_root, fname)
    print("Located tif:", tif_path)

    with rasterio.open(tif_path) as src:
        img = src.read()  # (bands, H, W)
        H, W = src.height, src.width

    poly_rows = df[df["filename"] == fname]
    print("Rows matched:", len(poly_rows))

    # Parse WKT polygons -> GeoJSON mappings for rasterize()
    shapes = []
    bounds = []
    for g in poly_rows["geometry"].tolist():
        geom = wkt.loads(g)
        bounds.append(geom.bounds)  # (minx, miny, maxx, maxy)
        shapes.append((mapping(geom), 1))

    # Quick bounds sanity check
    minx = min(b[0] for b in bounds); miny = min(b[1] for b in bounds)
    maxx = max(b[2] for b in bounds); maxy = max(b[3] for b in bounds)
    print(f"Image size: W={W}, H={H}")
    print(f"Poly bounds: x[{minx:.1f},{maxx:.1f}]  y[{miny:.1f},{maxy:.1f}]")

    # Pixel-space transform (x=col, y=row)
    pixel_transform = Affine.identity()

    mask = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=pixel_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    print("Mask sum:", int(mask.sum()), "Mask max:", int(mask.max()))

    out_dir = project_root / "runs" / "debug_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make quick RGB preview
    if img.shape[0] >= 3:
        r, g, b = img[2], img[1], img[0]
        rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        rgb = rgb - rgb.min()
        if rgb.max() > 0:
            rgb = rgb / rgb.max()
        rgb8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        band = img[0].astype(np.float32)
        band = band - band.min()
        if band.max() > 0:
            band = band / band.max()
        rgb8 = (band * 255).astype(np.uint8)

    import imageio.v2 as imageio
    imageio.imwrite(out_dir / "image.png", rgb8)
    imageio.imwrite(out_dir / "mask.png", (mask * 255).astype(np.uint8))

    if rgb8.ndim == 3:
        overlay = rgb8.copy()
        overlay[mask == 1] = np.array([255, 0, 0], dtype=np.uint8)
        imageio.imwrite(out_dir / "overlay.png", overlay)

    print("Wrote:")
    print(" -", out_dir / "image.png")
    print(" -", out_dir / "mask.png")
    print(" -", out_dir / "overlay.png")

if __name__ == "__main__":
    main()
