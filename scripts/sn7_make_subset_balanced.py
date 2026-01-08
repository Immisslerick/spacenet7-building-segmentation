from pathlib import Path
import pandas as pd
import numpy as np
def main():
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "runs" / "sn7_manifest.csv"
    out_path = project_root / "runs" / "sn7_subset.csv"

    df = pd.read_csv(manifest_path)

    # ---- knobs: keep it manageable ----
    TOP_TILES = 12          # number of unique tiles
    MONTHS_PER_TILE = 6     # number of months/images per tile
    MIN_POLYS = 50        # drop nearly-empty images

    df = df[df["n_polygons"] >= MIN_POLYS].copy()

    SEED = 42
    rng = np.random.default_rng(SEED)

    
    N_DENSE  = 4
    N_MED    = 4
    N_SPARSE = 4
    assert N_DENSE + N_MED + N_SPARSE == TOP_TILES

    # total polygons per tile
    tile_totals = df.groupby("tile")["n_polygons"].sum().sort_values(ascending=False)

    # split tiles into thirds by density
    tiles_sorted = tile_totals.index.to_list()
    n = len(tiles_sorted)
    dense_band  = tiles_sorted[: n//3]
    med_band    = tiles_sorted[n//3 : 2*n//3]
    sparse_band = tiles_sorted[2*n//3 :]

    dense_tiles  = rng.choice(dense_band,  size=min(N_DENSE,  len(dense_band)),  replace=False).tolist()
    med_tiles    = rng.choice(med_band,    size=min(N_MED,    len(med_band)),    replace=False).tolist()
    sparse_tiles = rng.choice(sparse_band, size=min(N_SPARSE, len(sparse_band)), replace=False).tolist()

    tile_rank = dense_tiles + med_tiles + sparse_tiles
    print("Picked tiles:", {"dense": dense_tiles, "med": med_tiles, "sparse": sparse_tiles})    


    subset_rows = []
    for tile in tile_rank:
        dft = df[df["tile"] == tile].copy()
        dft = dft.sort_values("n_polygons", ascending=False)

        k = MONTHS_PER_TILE
        k_hi = k // 2
        k_lo = k - k_hi

        hi = dft.head(min(k_hi, len(dft)))
        lo = dft.tail(min(k_lo, len(dft)))

        pick = pd.concat([hi, lo], ignore_index=True).drop_duplicates(subset=["filename"])
        pick = pick.head(MONTHS_PER_TILE)  # safety
        subset_rows.append(pick)


    subset = pd.concat(subset_rows, ignore_index=True)

    subset.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    print("Tiles:", subset["tile"].nunique(), "Images:", len(subset))
    print(subset.sort_values(["tile", "month"]).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
