# Breast Ultrasound Segmentation

A polished version of a deep learning class project on breast ultrasound segmentation. The TensorFlow U-Net implementation compares binary cross-entropy with class-balanced focal loss on the BUSI dataset.

This revision corrects image and mask pairing, combines multiple lesion masks, separates validation from final testing, and adds reproducible experiments and automated tests. The [original class materials](original_project/) include the proposal, slides, report, and results CSV. The findings below come from the corrected rerun.

## Data and methods

Images are resized to 128 × 128 and normalized to [0, 1]. All masks belonging to an image are combined before nearest-neighbor resizing. Two duplicate images with conflicting annotations are excluded, leaving 778 images: 546 training, 116 validation, and 116 test.

Both experiments use a four-level U-Net with 16 base channels, batch normalization, ReLU, and a sigmoid output. Training uses Adam at 1e-4, batch size 16, seed 42, and 30 epochs. Validation Dice selects the checkpoint; predictions are thresholded at 0.5.

## Results

Run date: **2026-10-04**. Scores are per-image averages on the held-out test split.

| Loss | Test Dice | Test IoU | Tumor-image Dice |
|---|---:|---:|---:|
| Binary cross-entropy | 0.3895 | 0.3107 | 0.4602 |
| Focal loss, α=0.7 | 0.4547 | 0.3516 | 0.5391 |

Focal loss produced higher overlap in this single-seed comparison. Both models predicted foreground on 19 of 20 normal test images. Splits are image-level; patient-level separation and external validation were not established.

[Findings](results/FINDINGS.md) · [Experiment protocol](docs/METHODS.md) · [Dataset audit](results/DATA_AUDIT.md)

![Test results](results/comparison.png)

## Run

Use Python 3.12, from the repository directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/download_data.py
python scripts/run_study.py
python scripts/evaluate_study.py
python scripts/write_findings.py
```

Completed runs are preserved. Use new run directories to repeat training. Raw data and checkpoints are stored locally under `data/` and `runs/`.

## Predict

```bash
python -m busi_segmentation.predict \
  --run runs/focal07_seed42 \
  --image /path/to/ultrasound.png \
  --output runs/prediction
```

The output contains a probability map, binary mask, and overlay. The mask is predicted at 128 × 128; its original-size version is rescaled.

## References

- [BUSI dataset](https://doi.org/10.1016/j.dib.2019.104863)
- [U-Net](https://arxiv.org/abs/1505.04597)
- [Attention U-Net](https://arxiv.org/abs/1804.03999)

Code: [MIT license](LICENSE). Dataset terms are separate.
