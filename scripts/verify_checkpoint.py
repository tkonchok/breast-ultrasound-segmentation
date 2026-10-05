"""Verify saved validation scores and inference batch invariance after reloading weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
from busi_segmentation.model import unet_model
from busi_segmentation.cli import evaluate_rows
from busi_segmentation.metrics import summarize

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--data-root', type=Path, required=True)
args = parser.parse_args()
config = json.loads((args.run / 'config.json').read_text())
manifest = json.loads((args.run / 'manifest.json').read_text())
model = unet_model(base_filters=config['base_filters'], activation=config['activation'],
                   residual=config['residual'], attention=config['attention'])
model.load_weights(args.run / 'best.weights.h5')
records = manifest['splits']['val']
a = summarize(evaluate_rows(model, args.data_root, records, 16))
b = summarize(evaluate_rows(model, args.data_root, records, 7))
saved = json.loads((args.run / 'validation.json').read_text())
for group in ['all_images', 'tumor_images']:
    for metric in ['dice', 'iou', 'pixel_accuracy']:
        np.testing.assert_allclose(a[group][metric], saved[group][metric], atol=1e-6, rtol=0)
        np.testing.assert_allclose(a[group][metric], b[group][metric], atol=1e-4, rtol=0)
result = {'checkpoint_reload_matches_saved_validation': True,
          'validation_batch16_matches_batch7': True, 'validation_images': len(records),
          'all_image_dice': a['all_images']['dice'], 'tumor_image_dice': a['tumor_images']['dice']}
(args.run / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
