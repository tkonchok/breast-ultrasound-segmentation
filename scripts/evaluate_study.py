"""Freeze validation decisions, then evaluate the two prespecified configurations."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parents[1]
os.chdir(project)
os.environ.setdefault('MPLCONFIGDIR', str(project / '.cache' / 'matplotlib'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
plan_path = project / 'configs/study.json'
plan = json.loads(plan_path.read_text())
source = json.loads((project / 'data/source.json').read_text())
lines = ['# Frozen final-evaluation record', '',
         'Both configurations were prespecified in `configs/study.json` before full training.',
         'Checkpoints were selected on validation all-image macro Dice. Threshold remains 0.5.',
         'No configuration is selected or tuned using test scores.', '',
         '| Run | Validation Dice | Tumor validation Dice | Best epoch |',
         '|---|---:|---:|---:|']
for experiment in plan['experiments']:
    run = project / 'runs' / experiment['name']
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads((run / 'training_summary.json').read_text())
    val = json.loads((run / 'validation.json').read_text())
    for key in ['loss', 'activation', 'residual', 'attention']:
        if config[key] != experiment[key]:
            raise ValueError(f'Run differs from the frozen experiment plan: {key}')
    for key in ['epochs', 'batch_size', 'base_filters']:
        if config[key] != plan[key]:
            raise ValueError(f'Run differs from the frozen budget: {key}')
    if config['seed'] != plan['training_seeds'][0]:
        raise ValueError('Training seed differs from the plan')
    lines.append(f"| {experiment['name']} | {val['all_images']['dice']:.6f} | {val['tumor_images']['dice']:.6f} | {summary['best_epoch']} |")
selection = project / 'results/selection.md'
if selection.exists() and selection.read_text() != '\n'.join(lines) + '\n':
    raise ValueError('An existing final-evaluation record differs; do not silently rewrite it')
selection.write_text('\n'.join(lines) + '\n')
(project / 'results/protocol.json').write_text(json.dumps({'study': plan,
    'study_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest()}, indent=2) + '\n')
for experiment in plan['experiments']:
    run = project / 'runs' / experiment['name']
    if (run / 'test_metrics.json').exists():
        print(f'Preserving prior test results: {run}', flush=True)
        continue
    subprocess.run([sys.executable, '-m', 'busi_segmentation.cli', 'evaluate',
                    '--data-root', source['local_root'], '--run', str(run)], check=True)
subprocess.run([sys.executable, 'scripts/export_results.py'], check=True)
