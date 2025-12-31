# SpaceNet 7 — Building Segmentation (Portfolio)

This repo documents a baseline building-footprint segmentation pipeline using SpaceNet 7 imagery.

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

## How to reproduce
1) Create venv + install deps  
2) Build manifest / masks / split  
3) Train baseline  
4) Generate prediction gallery
