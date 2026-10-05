"""Prepare data, train on train/validation only, and evaluate a frozen run."""
import argparse
import csv
import json
from pathlib import Path
import platform
import hashlib
import subprocess
import sys
import numpy as np
from .data import prepare, load_sample
from .metrics import score_mask, summarize


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def arrays(root, records):
    samples = [load_sample(root, record) for record in records]
    return np.stack([s[0] for s in samples]), np.stack([s[1] for s in samples])


def evaluate_rows(model, root, records, batch_size):
    # Collect individual scores; aggregation is independent of batch size.
    rows = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        x, y = arrays(root, chunk)
        predictions = model.predict(x, verbose=0)
        for record, truth, pred in zip(chunk, y, predictions):
            rows.append({'id': record['id'], 'category': record['category'], **score_mask(truth, pred)})
    return rows


def train(args, manifest):
    import tensorflow as tf
    from .model import unet_model, dice_loss, iou_loss, hard_dice_per_image
    run = Path(args.run)
    if run.exists():
        raise ValueError('Run directory already exists. Choose a new run path to preserve evidence.')
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    losses = {'bce': 'binary_crossentropy', 'dice': dice_loss, 'iou': iou_loss,
              'focal': tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=args.alpha, gamma=2)}

    def augment(image, mask):
        joined = tf.concat([image, mask], axis=-1)
        joined = tf.image.random_flip_left_right(joined)
        joined = tf.image.random_flip_up_down(joined)
        joined = tf.image.rot90(joined, tf.random.uniform((), maxval=4, dtype=tf.int32))
        return joined[..., :1], joined[..., 1:]

    x, y = arrays(args.data_root, manifest['splits']['train'])
    vx, vy = arrays(args.data_root, manifest['splits']['val'])
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=args.seed)
    if not args.no_augmentation:
        # Serial mapping makes the seeded random augmentation order explicit.
        ds = ds.map(augment)
    ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val = tf.data.Dataset.from_tensor_slices((vx, vy)).batch(16)
    model = unet_model(input_shape=(128, 128, 1), residual=args.residual,
                       attention=args.attention, activation=args.activation, base_filters=args.base_filters)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=losses[args.loss],
                  metrics=[tf.keras.metrics.MeanMetricWrapper(hard_dice_per_image, name='dice'),
                           tf.keras.metrics.BinaryAccuracy(name='pixel_accuracy')])
    run.mkdir(parents=True)
    write_json(run / 'config.json', config)
    write_json(run / 'manifest.json', manifest)
    write_json(run / 'source_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in Path(__file__).parent.glob('*.py')})
    write_json(run / 'environment.json', {'python': platform.python_version(), 'tensorflow': tf.__version__,
                                         'numpy': np.__version__, 'platform': platform.platform(),
                                         'devices': [str(d) for d in tf.config.list_physical_devices()]})
    (run / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    checkpoint = run / 'best.weights.h5'
    history = model.fit(ds, validation_data=val, epochs=args.epochs, verbose=2, shuffle=False, callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_dice', mode='max', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_dice', mode='max', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dice', mode='max', patience=5, factor=.5, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(str(run / 'history.csv'))])
    optimizer_updates = int(model.optimizer.iterations.numpy())
    if any(not np.isfinite(values).all() for values in history.history.values()):
        raise ValueError('Training produced nonfinite values; this run is not valid')
    model.load_weights(checkpoint)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ('loss', 'dice')):
        ax.plot(history.history[metric], label='train')
        ax.plot(history.history['val_' + metric], label='validation')
        ax.set_xlabel('Epoch (zero indexed)')
        ax.set_ylabel(metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(run / 'training_curves.png', dpi=150)
    plt.close(fig)
    write_json(run / 'training_summary.json', {'epochs_completed': len(history.epoch), 'parameters': model.count_params(),
                                               'optimizer_updates': optimizer_updates,
                                               'best_epoch': int(np.argmax(history.history['val_dice'])) + 1,
                                               'checkpoint_metric': 'validation all-image macro Dice'})
    write_json(run / 'validation.json', summarize(evaluate_rows(model, args.data_root, manifest['splits']['val'], 16)))
    print(f'Training complete. Review {run}/validation.json. Test data has not been evaluated.')


def evaluate(args, manifest):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    from .model import unet_model
    run = Path(args.run)
    config = json.loads((run / 'config.json').read_text())
    saved = json.loads((run / 'manifest.json').read_text())
    if manifest != saved:
        raise ValueError('Current data manifest differs from the training manifest')
    if (run / 'test_metrics.json').exists():
        raise ValueError('Test results already exist; preserve them rather than repeatedly probing test performance')
    model = unet_model(input_shape=(128, 128, 1), residual=config['residual'],
                       attention=config['attention'], activation=config['activation'], base_filters=config['base_filters'])
    model.load_weights(run / 'best.weights.h5')
    rows = evaluate_rows(model, args.data_root, manifest['splits']['test'], args.batch_size)
    with (run / 'test_per_image.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    empty = []
    for record in manifest['splits']['test']:
        _, truth = load_sample(args.data_root, record)
        empty.append({'category': record['category'], **score_mask(truth, np.zeros_like(truth))})
    write_json(run / 'test_metrics.json', {'threshold': .5, 'model': summarize(rows), 'all_background_baseline': summarize(empty)})
    # Show both successes and failures among tumor cases, determined by per-image Dice.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ranked = sorted((r for r in rows if r['category'] != 'normal'), key=lambda r: r['dice'])
    chosen = ranked[:3] + ranked[-3:]
    lookup = {r['id']: r for r in manifest['splits']['test']}
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)), squeeze=False)
    for i, row in enumerate(chosen):
        image, mask = load_sample(args.data_root, lookup[row['id']])
        pred = model.predict(image[None], verbose=0)[0] > .5
        for ax, value, title in zip(axes[i], (image, mask, pred), ('Ultrasound', 'Ground truth', 'Prediction')):
            ax.imshow(value.squeeze(), cmap='gray'); ax.set_title(title); ax.axis('off')
        axes[i, 0].set_title(f'{row["id"]}\nDice={row["dice"]:.3f}', fontsize=8)
    fig.tight_layout(); fig.savefig(run / 'test_examples.png', dpi=150); plt.close(fig)
    print(f'Test evaluation saved in {run}. Use these results for reporting; tune only on validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--data-root', type=Path, required=True, help='Folder containing benign/, malignant/, normal/')
    parser.add_argument('--manifest', type=Path, default=Path('results/splits.json'))
    parser.add_argument('--exclusions', type=Path, default=Path('configs/data_exclusions.json'), help='Audited BUSI exclusion rules')
    parser.add_argument('--run', type=Path, default=Path('runs/baseline_seed42'))
    parser.add_argument('--seed', type=int, default=42, help='Training seed; split seed is fixed at 42')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--base-filters', type=int, default=16, help='16 for compact U-Net; 64 for original width')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--loss', choices=['bce', 'focal', 'dice', 'iou'], default='bce')
    parser.add_argument('--alpha', type=float, default=.7)
    parser.add_argument('--activation', choices=['relu', 'swish'], default='relu')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--no-augmentation', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.base_filters < 2 or args.cpu_threads < 1 or not 0 <= args.alpha <= 1:
        parser.error('Batch size, epochs, and thread count must be positive; base filters >= 2; alpha in [0,1]')
    manifest = prepare(args.data_root, args.manifest, args.exclusions)
    if args.command == 'prepare':
        print({name: len(records) for name, records in manifest['splits'].items()})
    elif args.command == 'train':
        train(args, manifest)
    else:
        evaluate(args, manifest)


if __name__ == '__main__':
    main()
