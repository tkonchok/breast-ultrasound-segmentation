"""Run a saved segmenter on one image and export its mask and overlay."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for prediction artifacts')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory exists; choose a new path')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    from .model import unet_model
    config = json.loads((args.run / 'config.json').read_text())
    model = unet_model(residual=config['residual'], attention=config['attention'],
                       activation=config['activation'], base_filters=config['base_filters'])
    model.load_weights(args.run / 'best.weights.h5')
    with Image.open(args.image) as source:
        gray = source.convert('L')
        image = np.asarray(gray.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        original = np.asarray(gray.convert('RGB'), dtype=np.float32)
    probabilities = model(image[None, ..., None], training=False).numpy()[0, ..., 0]
    mask = Image.fromarray((probabilities > .5).astype(np.uint8) * 255)
    scaled_mask = mask.resize(gray.size, Image.Resampling.NEAREST)
    foreground = np.asarray(scaled_mask) > 0
    overlay = original.copy()
    overlay[foreground] = .6 * overlay[foreground] + .4 * np.array([255, 0, 0])
    args.output.mkdir(parents=True)
    mask.save(args.output / 'mask_128.png')
    scaled_mask.save(args.output / 'mask_original_size.png')
    Image.fromarray(overlay.astype(np.uint8)).save(args.output / 'overlay.png')
    np.save(args.output / 'probabilities.npy', probabilities)
    (args.output / 'prediction.json').write_text(json.dumps({'image': str(args.image), 'run': str(args.run),
        'threshold': .5, 'model_input_size': [128, 128], 'original_size': list(gray.size),
        'foreground_pixels_128': int((probabilities > .5).sum())}, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
