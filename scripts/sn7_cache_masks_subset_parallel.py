from pathlib import Path
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import rasterio
from rasterio.features import rasterize
from affine import Affine
from shapely import wkt
from shapely.geometry import mapping
import imageio.v2 as imageio

PIXEL_TRANSFORM = Affine.identity()

def _rasterize_one(tif_path: str, wkts: list[str], out_path: str) -> tuple[str, str]:
    out_path = Path(out_path)
    if out_path.exists():
        return (str(out_path), "skipped")

    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width

    shapes = [(mapping(wkt.loads(g)), 1) for g in wkts]

    mask = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=PIXEL_TRANSFORM,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, (mask * 255).astype(np.uint8))
    return (str(out_path), "ok")

def main():
    project_root = Path(__file__).resolve().parent.parent
    labels_csv = project_root / "data" / "sn7" / "SN7_buildings_train_csvs" / "csvs" / "sn7_train_ground_truth_pix.csv"
    subset_csv = project_root / "runs" / "sn7_subset.csv"

    df_labels = pd.read_csv(labels_csv)
    subset = pd.read_csv(subset_csv)

    # Build WKT lists only for this subset
    needed = set(subset["filename"].tolist())
    df_small = df_labels[df_labels["filename"].isin(needed)]
    wkt_map = df_small.groupby("filename")["geometry"].apply(list).to_dict()

    out_dir = project_root / "data" / "processed" / "masks_png"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, row in subset.iterrows():
        stem = row["filename"]     # CSV stem (no .tif)
        tif_path = row["tif_path"]
        wkts = wkt_map.get(stem, [])
        if not wkts:
            continue
        out_path = out_dir / f"{stem}_mask.png"
        tasks.append((tif_path, wkts, str(out_path)))

    print(f"Mask cache tasks: {len(tasks)}")

    # i9-9900K: start at 6; bump to 8 if stable
    max_workers = min(6, os.cpu_count() or 6)
    print("Workers:", max_workers)

    ok = skipped = failed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_rasterize_one, *t) for t in tasks]
        for f in as_completed(futures):
            try:
                _, status = f.result()
                ok += (status == "ok")
                skipped += (status == "skipped")
            except Exception as e:
                failed += 1
                print("FAILED:", repr(e))

    print(f"Done. ok={ok}, skipped={skipped}, failed={failed}")
    print("Masks in:", out_dir)

if __name__ == "__main__":
    main()
