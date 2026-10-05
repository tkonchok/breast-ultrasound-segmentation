# Frozen final-evaluation record

Both configurations were prespecified in `configs/study.json` before full training.
Checkpoints were selected on validation all-image macro Dice. Threshold remains 0.5.
No configuration is selected or tuned using test scores.

| Run | Validation Dice | Tumor validation Dice | Best epoch |
|---|---:|---:|---:|
| baseline_bce_seed42 | 0.370377 | 0.447539 | 28 |
| focal07_seed42 | 0.491667 | 0.583681 | 29 |
