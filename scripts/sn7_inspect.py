from pathlib import Path
import pandas as pd

def count_exts(root: Path):
    exts = [".tif", ".tiff", ".png", ".jpg", ".geojson", ".json", ".csv"]
    counts = {e: 0 for e in exts}
    for p in root.rglob("*"):
        if p.is_file():
            s = p.suffix.lower()
            if s in counts:
                counts[s] += 1
    return counts

def first_file(root: Path, ext: str):
    for p in root.rglob(f"*{ext}"):
        return p
    return None

def main():
    # scripts/ is a sibling of data/, so go up one level
    project_root = Path(__file__).resolve().parent.parent
    root = project_root / "data" / "sn7"

    if not root.exists():
        print(f"[!] Not found: {root}")
        print("Expected: <project>/data/sn7/...")
        return

    print("Project root:", project_root)
    print("Data root:   ", root)

    print("\n=== Top folders ===")
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        print("-", p.name)

    print("\n=== Extension counts (overall) ===")
    counts = count_exts(root)
    for k, v in counts.items():
        print(f"{k:8s} {v}")

    csv_dir = root / "SN7_buildings_train_csvs"
    csv = first_file(csv_dir, ".csv")
    print("\n=== Sample CSV ===")
    if csv:
        print("CSV:", csv)
        df = pd.read_csv(csv)
        print("Columns:", list(df.columns))
        print(df.head(3))
    else:
        print(f"No CSV found in: {csv_dir}")

if __name__ == "__main__":
    main()
