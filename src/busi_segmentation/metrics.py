"""Per-image foreground scores; empty/empty scores 1, empty/nonempty scores 0."""
import numpy as np


def score_mask(truth, probabilities, threshold=.5):
    truth = np.asarray(truth)
    probabilities = np.asarray(probabilities)
    if truth.shape != probabilities.shape or not truth.size:
        raise ValueError('Masks must have the same, nonempty shape')
    if not np.isfinite(truth).all() or not np.isfinite(probabilities).all() or not 0 <= threshold <= 1:
        raise ValueError('Masks must be finite and threshold in [0,1]')
    if np.any((probabilities < 0) | (probabilities > 1)) or not np.isin(truth, [0, 1]).all():
        raise ValueError('Ground truth must be binary and probabilities in [0,1]')
    truth = truth > .5
    pred = probabilities > threshold
    intersection = int(np.count_nonzero(truth & pred))
    total = int(truth.sum() + pred.sum())
    union = total - intersection
    return {'dice': 2 * intersection / total if total else 1.,
            'iou': intersection / union if union else 1.,
            'pixel_accuracy': float(np.mean(truth == pred)),
            'true_pixels': int(truth.sum()), 'predicted_pixels': int(pred.sum())}


def summarize(rows):
    if not rows:
        raise ValueError('Cannot summarize an empty evaluation')
    def mean(group):
        return {'n': len(group), **{key: float(np.mean([r[key] for r in group])) if group else None
                                  for key in ('dice', 'iou', 'pixel_accuracy')}}
    normal = [r for r in rows if r['category'] == 'normal']
    return {'all_images': mean(rows),
            'tumor_images': mean([r for r in rows if r['category'] != 'normal']),
            'by_category': {c: mean([r for r in rows if r['category'] == c])
                            for c in ('benign', 'malignant', 'normal')},
            'normal_false_positive_image_rate': float(np.mean([r['predicted_pixels'] > 0 for r in normal])) if normal else None}
