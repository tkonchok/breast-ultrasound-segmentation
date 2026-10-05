"""Download a pinned BUSI mirror version into the project, not the user's global cache."""
import json
import os
from pathlib import Path

project = Path(__file__).resolve().parents[1]
os.environ['KAGGLEHUB_CACHE'] = str(project / 'data' / 'cache')
import kagglehub

handle = 'aryashah2k/breast-ultrasound-images-dataset/versions/1'
path = Path(kagglehub.dataset_download(handle))
root = path / 'Dataset_BUSI_with_GT'
if not root.is_dir():
    raise FileNotFoundError(f'Dataset layout differs: {path}')
record = {'handle': handle, 'local_root': str(root), 'kagglehub_version': kagglehub.__version__}
(project / 'data' / 'source.json').write_text(json.dumps(record, indent=2) + '\n')
print(root)
