from pathlib import Path
import pandas as pd
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/sn7_check_labels_for_tif.py <FULL_PATH_TO_TIF>")
        return

    tif_path = Path(sys.argv[1])
    if not tif_path.exists():
        print(f"[!] File not found: {tif_path}")
        return

    csv_path = Path("data/sn7/SN7_buildings_train_csvs/csvs/sn7_train_ground_truth_pix.csv")
    if not csv_path.exists():
        print(f"[!] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    name = tif_path.name

    print("TIF:", tif_path)
    print("Filename key:", name)

    exact = (df["filename"] == name)
    print("Exists in CSV exactly?:", bool(exact.any()))

    if exact.any():
        print("Polygon rows:", int(exact.sum()))
        return

    stem = tif_path.stem[:35]
    hits = df[df["filename"].str.contains(stem, na=False)]["filename"].head(10)

    print("\nNear matches (first 10):")
    if len(hits) == 0:
        print("  (none)")
    else:
        for h in hits.tolist():
            print(" ", h)

if __name__ == "__main__":
    main()
