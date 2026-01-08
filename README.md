# SpaceNet 7 — Building Segmentation (Portfolio)

This repo documents a baseline building-footprint segmentation pipeline using SpaceNet 7 imagery.

## Progress log (living notes)

This section is updated after each “official run” so the repo shows iterative progress.

### Current best (so far)
- **Best validation IoU (crop-based):** ~0.2418 @ **thr=0.40** (CPU run, tile-based split)
- **Model:** TinyUNet (U-Net style) **in_ch=4**
- **Notes:** Threshold matters; IoU varies across thresholds. See threshold sweep output in scripts.

### Runs summary

| Run ID | Date | Model | Train/Val split | Val IoU (best thr) | Precision/Recall (snapshot) | Notes |
|---|---:|---|---|---:|---|---|
| baseline_v1 | (prior) | TinyUNet (base=32) | (older setup) | 0.2382 | — | Baseline v1 metrics + gallery committed under `results/baseline_v1/` |
| server_run_v2 | 2026-01-08 | TinyUNet (base=64) | tile split (`sn7_split.csv`) | ~0.2418 @0.40 | e.g. @0.40 P~0.3996 R~0.4510 F1~0.4238 | Added PR metrics + subset/split workflow; CPU training |

### What changed in server_run_v2
- Added/updated: threshold sweep + PR metrics reporting
- Added/updated: subset selection (`sn7_make_subset_balanced.py`) and tile split (`sn7_split_by_tile.py`)
- Updated eval/pred scripts to match checkpoint architecture (**base must match**)

### Next planned improvements
- Improve data sampling: more “positive” crops early (building-heavy chips)
- Try Dice/Focal loss mix (or BCE + Dice) to combat class imbalance
- Increase spatial context (larger crops or multi-scale) / light augmentations
- Validate on full tiles (not only random crops) for a more honest metric



## Baseline results
| Experiment | Model | Device | Val IoU |
|---|---|---:|---:|
| baseline_v1 | TinyUNet (base=32) | CPU | 0.2382 |

See qualitative examples: `results/baseline_v1/gallery/`  
See metrics: `results/baseline_v1/metrics.md`

## Repo layout
- `scripts/` training + utility scripts
- `results/` small, committed artifacts for portfolio
- `runs/` large run outputs (NOT committed)
- `Data/` raw dataset (NOT committed)

## Workflow (scripts)

Typical order for an “official run”:

1) (If needed) Build/update mask cache (writes to `data/processed/masks_png/`)
   - `scripts/sn7_cache_masks_parallel.py`
   - `scripts/sn7_verify_cached_masks.py`

2) Build manifest / dataset index (writes under `runs/`)
   - `scripts/sn7_make_manifest.py`

3) Build a manageable subset (writes `runs/sn7_subset.csv`)
   - `scripts/sn7_make_subset_balanced.py`

4) Tile-based split (writes `runs/sn7_split.csv`)
   - `scripts/sn7_split_by_tile.py`

5) Train baseline model (writes checkpoint under `runs/`)
   - `scripts/sn7_train_baseline.py`

6) Evaluate thresholds / metrics
   - `scripts/sn7_eval_thresholds.py`

7) Visual sanity check (gallery)
   - `scripts/sn7_pred_gallery.py`

Artifacts policy:
- `results/` = small portfolio artifacts committed (metrics, images, summaries)
- `runs/` and `data/` = local/large outputs NOT committed


## How to reproduce
1) Create venv + install deps  
2) Build manifest / masks / split  
3) Train baseline  
4) Generate prediction gallery
