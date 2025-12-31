# Baseline v1 — TinyUNet (CPU)

## Run summary
- Device: CPU
- DataLoader workers: 8
- Input channels: 4
- Epochs: 6
- Batch size: 4
- Crop: 512
- Train samples/epoch: 2500
- Val samples/epoch: 400

## Final metrics
Epoch 6/6  train_loss=0.3379  val_IoU=0.2382  epoch_min=26.7

## Artifacts
- Weights (not committed): `runs/baseline_unet.pt`
- Example overlays: `results/baseline_v1/gallery/`

## Notes
- Predictions are conservative (misses many buildings), consistent with lower recall and modest IoU.
