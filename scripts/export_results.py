"""Export reviewed evidence without raw images or trained weights."""
import csv
import json
from pathlib import Path
import shutil

project = Path(__file__).resolve().parents[1]
plan = json.loads((project / 'configs/study.json').read_text())
rows = []
for experiment in plan['experiments']:
    name = experiment['name']
    run = project / 'runs' / name
    metrics = json.loads((run / 'test_metrics.json').read_text())
    val = json.loads((run / 'validation.json').read_text())
    summary = json.loads((run / 'training_summary.json').read_text())
    target = project / 'results' / name
    target.mkdir(exist_ok=True)
    for filename in ['config.json', 'environment.json', 'source_hashes.json', 'pip-freeze.txt',
                     'history.csv', 'training_summary.json', 'validation.json',
                     'test_per_image.csv', 'test_metrics.json', 'training_curves.png', 'verification.json']:
        if filename == 'verification.json' and not (run / filename).exists():
            continue
        if filename == 'config.json':
            config = json.loads((run / filename).read_text())
            for key in ['data_root', 'run', 'manifest', 'exclusions']:
                config[key] = {'data_root': 'data/cache/datasets/aryashah2k/breast-ultrasound-images-dataset/versions/1/Dataset_BUSI_with_GT',
                               'run': 'runs/' + name, 'manifest': 'results/splits.json',
                               'exclusions': 'configs/data_exclusions.json'}[key]
            (target / filename).write_text(json.dumps(config, indent=2) + '\n')
        elif filename == 'pip-freeze.txt':
            lines=(run / filename).read_text().splitlines()
            (target / filename).write_text('\n'.join(line for line in lines if not line.startswith(('-e ', '# Editable'))) + '\n')
        else:
            shutil.copy2(run / filename, target / filename)
    # Dataset-derived example figures stay in runs/ until redistribution terms permit them.
    row = {'experiment': name, 'validation_dice': val['all_images']['dice'],
           'test_dice': metrics['model']['all_images']['dice'], 'test_iou': metrics['model']['all_images']['iou'],
           'tumor_test_dice': metrics['model']['tumor_images']['dice'],
           'tumor_test_iou': metrics['model']['tumor_images']['iou'],
           'pixel_accuracy': metrics['model']['all_images']['pixel_accuracy'],
           'normal_false_positive_image_rate': metrics['model']['normal_false_positive_image_rate'],
           'epochs_completed': summary['epochs_completed'], 'best_epoch': summary['best_epoch'],
           'parameters': summary['parameters'], 'optimizer_updates': summary['optimizer_updates']}
    rows.append(row)
with (project / 'results/comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
background = metrics['all_background_baseline']
(project / 'results/all_background_baseline.json').write_text(json.dumps(background, indent=2) + '\n')
print(json.dumps(rows, indent=2))
