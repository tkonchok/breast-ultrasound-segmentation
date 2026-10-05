# Experiment protocol

## Dataset

Use `aryashah2k/breast-ultrasound-images-dataset/versions/1`. Match masks by the exact image stem and combine all annotations. Exclude both members of the ambiguous duplicate pair recorded in `configs/data_exclusions.json`. The raw dataset is preserved; exclusion rules verify decoded-image hashes.

The retained dataset has 436 benign, 209 malignant, and 133 normal images. Stratified splits use seed 42: 546 train, 116 validation, and 116 test. Exact records and annotation hashes are saved in `results/splits.json`. Patient identities are unavailable, and near duplicates were not exhaustively checked.

Images use grayscale bilinear resizing to 128 × 128. Masks are unioned at native resolution and resized with nearest-neighbor interpolation. Training augmentation applies the same flips and quarter-turn rotations to each image and mask.

## Training

Both configurations use encoder widths 16/32/64/128, a 256-channel bottleneck, batch normalization, ReLU, skip connections, and a sigmoid output. The model has 1,946,705 parameters. Residual blocks, attention gates, Swish, and alternate overlap losses are available in the source; the reported comparison trains BCE and focal loss only.

The fixed study is in `configs/study.json`: seed 42, batch 16, Adam at 1e-4, and 30 epochs. Focal loss uses class balancing with alpha 0.7 and gamma 2. TensorFlow deterministic operations are enabled.

Select the checkpoint using validation all-image macro Dice. Early stopping uses patience 15. Learning-rate reduction uses patience 5, factor 0.5, and minimum rate 1e-6. Each completed run records its configuration, environment, source hashes, history, update count, and selected epoch.

## Evaluation

Threshold foreground probabilities at `> 0.5`. Compute per-image Dice and IoU, then average images rather than batch averages. Both-empty masks receive 1; one-empty masks receive 0.

Report all-image overlap, benign/malignant-image overlap, category scores, pixel accuracy, and the fraction of normal images with any foreground prediction. Include an all-background reference. Test data is evaluated after validation checkpoint selection; it does not tune the model or threshold.

`results/selection.md` records checkpoint decisions. Per-image test scores, category summaries, and training histories are retained in `results/`. Prediction figures are generated locally under `runs/`.

## Verification

Run the 13 automated tests with `python -m unittest discover -s tests -v`. They check mask unions, dataset errors, exclusion hashes, split reproducibility, metric aggregation, model variants, finite gradients, and checkpoint reloading.

To check a trained checkpoint against its saved validation results and compare inference batches of 16 and 7:

```bash
python scripts/verify_checkpoint.py \
  --run runs/focal07_seed42 \
  --data-root data/cache/datasets/aryashah2k/breast-ultrasound-images-dataset/versions/1/Dataset_BUSI_with_GT
```

The local runs used Python 3.12.14 and TensorFlow 2.20.0 on macOS arm64 CPU. This is a single-seed experiment on one dataset, not a statistically established performance ranking.
