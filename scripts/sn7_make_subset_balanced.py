from pathlib import Path
import pandas as pd

def main():
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "runs" / "sn7_manifest.csv"
    out_path = project_root / "runs" / "sn7_subset.csv"

    df = pd.read_csv(manifest_path)

    # ---- knobs: keep it manageable ----
    TOP_TILES = 12          # number of unique tiles
    MONTHS_PER_TILE = 6     # number of months/images per tile
    MIN_POLYS = 500         # drop nearly-empty images

    df = df[df["n_polygons"] >= MIN_POLYS].copy()

    # rank tiles by total polygons across months
    tile_rank = (
        df.groupby("tile")["n_polygons"].sum()
          .sort_values(ascending=False)
          .head(TOP_TILES)
          .index
          .tolist()
    )

    subset_rows = []
    for tile in tile_rank:
        dft = df[df["tile"] == tile].sort_values("n_polygons", ascending=False)
        subset_rows.append(dft.head(MONTHS_PER_TILE))

    subset = pd.concat(subset_rows, ignore_index=True)

    subset.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    print("Tiles:", subset["tile"].nunique(), "Images:", len(subset))
    print(subset.sort_values(["tile", "month"]).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
