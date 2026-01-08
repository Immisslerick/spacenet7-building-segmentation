from pathlib import Path
import pandas as pd
import random

def main():
    project_root = Path(__file__).resolve().parent.parent
    subset_path = project_root / "runs" / "sn7_subset.csv"
    out_path = project_root / "runs" / "sn7_split_subset.csv"


    df = pd.read_csv(subset_path)

    tiles = sorted(df["tile"].unique().tolist())
    random.seed(42)
    random.shuffle(tiles)

    # Hold out ~20% tiles for validation (at least 2)
    n_val = max(2, int(round(0.2 * len(tiles))))
    val_tiles = set(tiles[:n_val])

    df["split"] = df["tile"].apply(lambda t: "val" if t in val_tiles else "train")

    print("Train tiles:", df[df["split"]=="train"]["tile"].nunique(),
          "Val tiles:", df[df["split"]=="val"]["tile"].nunique())
    print("Train images:", (df["split"]=="train").sum(),
          "Val images:", (df["split"]=="val").sum())

    df.to_csv(out_path, index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
