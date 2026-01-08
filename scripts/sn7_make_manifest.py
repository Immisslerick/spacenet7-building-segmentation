from pathlib import Path
import pandas as pd

def locate_tif(train_root: Path, stem: str):
    # Match by stem, because CSV 'filename' has no .tif extension
    return next(train_root.rglob(f"{stem}.tif"), None) or next(train_root.rglob(f"{stem}*.tif"), None)

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data" / "sn7"
    train_root = data_root / "SN7_buildings_train" / "train"
    csv_path = data_root / "SN7_buildings_train_csvs" / "csvs" / "sn7_train_ground_truth_pix.csv"

    df = pd.read_csv(csv_path)

    manifest = df.groupby("filename").size().reset_index(name="n_polygons")
    manifest["month"] = manifest["filename"].str.extract(r"(global_monthly_\d{4}_\d{2})", expand=False)
    manifest["tile"]  = manifest["filename"].str.extract(r"(L\d{2}-\d{4}E-\d{4}N_\d+_\d+_\d+)", expand=False)

    # locate actual tif path
    manifest["tif_path"] = manifest["filename"].apply(lambda s: str(locate_tif(train_root, s)) if locate_tif(train_root, s) else None)
    manifest = manifest.dropna(subset=["tif_path"]).copy()

    out_dir = project_root / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sn7_manifest.csv"
    manifest.sort_values("n_polygons", ascending=False).to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print("Images with labels + tif found:", len(manifest))
    print(manifest.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
