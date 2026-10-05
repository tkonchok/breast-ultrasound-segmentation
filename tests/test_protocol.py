import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from busi_segmentation.data import discover, make_splits, load_sample, prepare
from busi_segmentation.metrics import score_mask, summarize


class ProtocolTests(unittest.TestCase):
    def fixture(self, root):
        for j, category in enumerate(('benign', 'malignant', 'normal')):
            folder = root / category
            folder.mkdir()
            for i in range(10):
                pixels = np.full((4, 4), i + j * 10, np.uint8)
                Image.fromarray(pixels).save(folder / f'{category} ({i}).png')
                Image.fromarray(np.zeros((4, 4), np.uint8)).save(folder / f'{category} ({i})_mask.png')

    def test_mask_union_and_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            mask = np.zeros((4, 4), np.uint8); mask[0, 0] = 255
            Image.fromarray(mask).save(root / 'benign/benign (0)_mask.png')
            mask[:] = 0; mask[3, 3] = 255
            Image.fromarray(mask).save(root / 'benign/benign (0)_mask_1.png')
            records = discover(root)
            r = next(r for r in records if r['id'] == 'benign/benign (0).png')
            image, union = load_sample(root, r, size=4)
            self.assertEqual(union.sum(), 2)
            self.assertEqual(image.shape, (4, 4, 1))
            manifest = make_splits(records)
            self.assertEqual(manifest, make_splits(records))
            ids = [r['id'] for split in manifest['splits'].values() for r in split]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {r['id'] for r in records})
            for split in manifest['splits'].values():
                self.assertEqual({r['category'] for r in split}, {'benign', 'malignant', 'normal'})
            self.assertEqual(prepare(root, root / 'splits.json'), prepare(root, root / 'splits.json'))
            Image.fromarray(np.ones((4, 4), np.uint8)).save(root / 'benign/benign (0)_mask.png')
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                prepare(root, root / 'splits.json')

    def test_missing_or_duplicate_data_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            (root / 'benign/benign (0)_mask.png').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing mask'):
                discover(root)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            (root / 'benign/benign (1).png').write_bytes((root / 'benign/benign (0).png').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                discover(root)

    def test_overlap_edge_cases(self):
        empty = np.zeros((2, 2)); full = np.ones((2, 2))
        self.assertEqual(score_mask(empty, empty)['dice'], 1)
        self.assertEqual(score_mask(full, empty)['dice'], 0)
        self.assertEqual(score_mask(empty, full)['iou'], 0)
        half = np.array([[1, 1], [0, 0]])
        self.assertAlmostEqual(score_mask(full, half)['dice'], 2/3)
        self.assertAlmostEqual(score_mask(full, half)['iou'], .5)

    def test_macro_aggregation_is_partition_invariant(self):
        rows = [{'category': 'benign', **score_mask(np.ones((2, 2)), np.zeros((2, 2)))} for _ in range(5)]
        rows.append({'category': 'normal', **score_mask(np.zeros((2, 2)), np.zeros((2, 2)))})
        by_batches = [r for chunk in (rows[:4], rows[4:]) for r in chunk]
        summary = summarize(rows)
        self.assertEqual(summary, summarize(by_batches))
        self.assertAlmostEqual(summary['all_images']['dice'], 1/6)
        self.assertEqual(summary['tumor_images']['dice'], 0)

    def test_annotation_errors_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            record = next(r for r in discover(root) if r['category'] == 'normal')
            Image.fromarray(np.ones((4, 4), np.uint8) * 255).save(root / record['masks'][0])
            with self.assertRaisesRegex(ValueError, 'foreground annotation'):
                load_sample(root, record)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            record = discover(root)[0]
            Image.fromarray(np.zeros((3, 3), np.uint8)).save(root / record['masks'][0])
            with self.assertRaisesRegex(ValueError, 'dimension mismatch'):
                load_sample(root, record)
            Image.fromarray(np.zeros((4, 4), np.uint8)).save(root / 'benign/orphan_mask.png')
            with self.assertRaisesRegex(ValueError, 'Orphan masks'):
                discover(root)

    def test_explicit_exclusion_is_hash_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            original = discover(root)
            record = original[0]
            excluded = {'id': record['id'], 'image_sha256': record['image_sha256'], 'reason': 'test fixture'}
            included = discover(root, [excluded])
            self.assertEqual(len(included), len(original) - 1)
            self.assertNotIn(record['id'], {r['id'] for r in included})
            excluded['image_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'Excluded image changed'):
                discover(root, [excluded])

    def test_prediction_rows_are_batch_invariant(self):
        from busi_segmentation.cli import evaluate_rows
        class AllBackground:
            def predict(self, images, verbose=0):
                return np.zeros_like(images)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            records = discover(root)
            a = evaluate_rows(AllBackground(), root, records, 7)
            b = evaluate_rows(AllBackground(), root, records, 16)
            self.assertEqual(a, b)
            self.assertEqual(len(a), len(records))

    def test_invalid_masks_rejected(self):
        with self.assertRaises(ValueError):
            score_mask(np.ones((2, 2)), np.ones((3, 3)))
        with self.assertRaises(ValueError):
            score_mask(np.ones((2, 2)), np.full((2, 2), float('nan')))
        with self.assertRaises(ValueError):
            score_mask(np.full((2, 2), float('nan')), np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            score_mask(np.ones((2, 2)), np.full((2, 2), 2.))


if __name__ == '__main__':
    unittest.main()
