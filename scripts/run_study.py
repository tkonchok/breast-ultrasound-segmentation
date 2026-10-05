"""Execute the prespecified corrected study; keep test evaluation separate."""
import json
import os
from pathlib import Path
import subprocess
import sys

project = Path(__file__).resolve().parents[1]
os.chdir(project)
os.environ.setdefault('MPLCONFIGDIR', str(project / '.cache' / 'matplotlib'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
source = json.loads((project / 'data/source.json').read_text())
plan = json.loads((project / 'configs/study.json').read_text())
for experiment in plan['experiments']:
    run = project / 'runs' / experiment['name']
    if (run / 'training_summary.json').exists() and (run / 'validation.json').exists():
        print(f'Preserving completed run: {run}', flush=True)
        continue
    command = [sys.executable, '-m', 'busi_segmentation.cli', 'train', '--data-root', source['local_root'],
               '--run', str(run), '--epochs', str(plan['epochs']), '--seed', str(plan['training_seeds'][0]),
               '--base-filters', str(plan['base_filters']), '--batch-size', str(plan['batch_size']),
               '--loss', experiment['loss'], '--activation', experiment['activation']]
    if experiment.get('alpha') is not None:
        command.extend(['--alpha', str(experiment['alpha'])])
    if experiment['residual']:
        command.append('--residual')
    if experiment['attention']:
        command.append('--attention')
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, check=True)
print('Training complete. Review validation logs before the separate final evaluation.', flush=True)
