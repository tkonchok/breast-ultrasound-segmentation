"""Write the release report and a pixel-free comparison chart from exported metrics."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

project = Path(__file__).resolve().parents[1]
with (project / 'results/comparison.csv').open() as f:
    rows = list(csv.DictReader(f))
bce = json.loads((project / 'results/baseline_bce_seed42/test_metrics.json').read_text())['model']
focal = json.loads((project / 'results/focal07_seed42/test_metrics.json').read_text())['model']
background = json.loads((project / 'results/all_background_baseline.json').read_text())
zero_counts = {}
for name in ['baseline_bce_seed42', 'focal07_seed42']:
    with (project / 'results' / name / 'test_per_image.csv').open() as f:
        per_image = list(csv.DictReader(f))
    zero_counts[name] = sum(float(r['dice']) == 0 for r in per_image if r['category'] != 'normal')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
labels = ['BCE', 'Focal α=0.7', 'All background']
models = [bce, focal, background]
x = np.arange(3)
for offset, key, title, color in [(-.18, 'all_images', 'All 116 test images', '#3467a8'),
                                  (.18, 'tumor_images', '96 tumor test images', '#d97832')]:
    vals = [m[key]['dice'] for m in models]
    bars = axes[0].bar(x + offset, vals, .34, label=title, color=color)
    axes[0].bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
axes[0].set_xticks(x, labels)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Per-image macro Dice')
axes[0].set_title('Final test overlap')
axes[0].legend(loc='upper right', fontsize=9)
vals = [100 * m['normal_false_positive_image_rate'] for m in models]
bars = axes[1].bar(labels, vals, color=['#3467a8', '#d97832', '#6c7785'], width=.55)
axes[1].bar_label(bars, fmt='%.0f%%', padding=3, fontsize=10)
axes[1].set_ylim(0, 110)
axes[1].set_ylabel('Normal images with any predicted foreground (%)')
axes[1].set_title('False positives on 20 normal test images')
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
fig.suptitle('U-Net loss comparison · seed 42 · image-level test split', fontsize=13)
fig.tight_layout()
fig.savefig(project / 'results/comparison.png', dpi=160)
plt.close(fig)
report = """# Findings

Run date: 2026-10-04. Environment: Python 3.12.14, TensorFlow 2.20.0, macOS arm64 CPU.

## Setup

The BUSI version-1 mirror contains 780 images. Excluding two conflicting duplicate records leaves 778 images: 546 train, 116 validation, and 116 test. All 18 secondary annotation files are included in mask unions.

Both runs use the same 1,946,705-parameter U-Net, seed 42, batch size 16, and Adam at 1e-4. Each completed 30 epochs and 1,050 updates. BCE and class-balanced focal loss (alpha 0.7, gamma 2) are the two training objectives. Validation Dice selected epoch 28 for BCE and epoch 29 for focal loss. The threshold is fixed at > 0.5.

## Test results

| Model | Dice | IoU | Tumor Dice | Tumor IoU | Pixel accuracy | Normal FP image rate |
|---|---:|---:|---:|---:|---:|---:|
"""
for label, metrics in zip(labels, models):
    report += f"| {label} | {metrics['all_images']['dice']:.4f} | {metrics['all_images']['iou']:.4f} | {metrics['tumor_images']['dice']:.4f} | {metrics['tumor_images']['iou']:.4f} | {metrics['all_images']['pixel_accuracy']:.4f} | {metrics['normal_false_positive_image_rate']:.0%} |\n"
report += """
Overall scores average 116 images; tumor scores average the 96 benign/malignant images. Empty ground truth and empty prediction score 1. The all-background reference therefore has nonzero overall overlap despite zero tumor localization.

![Test comparison](comparison.png)

## Observations

"""
delta = focal['all_images']['dice'] - bce['all_images']['dice']
report += f"Focal loss increased observed test Dice by {delta:.4f} absolute. The all-background reference achieved {background['all_images']['pixel_accuracy']:.2%} pixel accuracy with zero tumor Dice, demonstrating why pixel accuracy alone is insufficient.\n\n"
report += f"Both trained models predicted foreground on 19 of 20 normal images. Mean normal foreground area was {100*(1-bce['by_category']['normal']['pixel_accuracy']):.2f}% for BCE and {100*(1-focal['by_category']['normal']['pixel_accuracy']):.2f}% for focal loss. Focal loss improved tumor overlap but produced more foreground area on normal images.\n\n"
report += "## Category scores\n\n| Category | Images | BCE Dice | Focal Dice | BCE IoU | Focal IoU |\n|---|---:|---:|---:|---:|---:|\n"
for cat in ['benign', 'malignant', 'normal']:
    a, b = bce['by_category'][cat], focal['by_category'][cat]
    report += f"| {cat.title()} | {a['n']} | {a['dice']:.4f} | {b['dice']:.4f} | {a['iou']:.4f} | {b['iou']:.4f} |\n"
report += f"\nBCE had {zero_counts['baseline_bce_seed42']} zero-overlap tumor cases; focal loss had {zero_counts['focal07_seed42']}. Prediction figures show missed lesions, incorrect localization, disconnected foreground, and border artifacts. They are generated locally under runs/.\n"
report += """
## Limits

This is a single-seed comparison on one dataset. Splits are image-level, patient identities are unavailable, near duplicates may remain, and there is no external validation. Higher overlap in this run does not establish statistical superiority or clinical performance. Normal-image false positives remain a substantial failure mode.

Configurations, histories, per-image scores, environments, and checkpoint-verification results are retained alongside this report.
"""
(project / 'results/FINDINGS.md').write_text(report)
print('Findings and comparison chart saved.')
