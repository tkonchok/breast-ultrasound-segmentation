"""Strict image/mask pairing and persisted, stratified image-level splits."""
from pathlib import Path
import hashlib
import json
import random
import re
import numpy as np
from PIL import Image

CATEGORIES = ('benign', 'malignant', 'normal')


def discover(root, exclusions=None):
    root = Path(root)
    exclusions = exclusions or []
    excluded = {r['id']: r for r in exclusions}
    verified_exclusions = set()
    records, hashes = [], {}
    for category in CATEGORIES:
        folder = root / category
        if not folder.is_dir():
            raise ValueError(f'Missing category directory: {folder}')
        images = sorted(p for p in folder.glob('*.png') if '_mask' not in p.stem)
        if len(images) < 7:
            raise ValueError(f'Need at least 7 images per category: {category}')
        used_masks = set()
        for image in images:
            pattern = re.compile(re.escape(image.stem) + r'_mask(?:_\d+)?\.png$')
            masks = sorted(p for p in folder.glob('*.png') if pattern.fullmatch(p.name))
            if not masks:
                raise ValueError(f'Missing mask for {image}')
            used_masks.update(masks)
            with Image.open(image) as source:
                pixels = np.asarray(source.convert('L'))
            digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            identity = f'{category}/{image.name}'
            if identity in excluded:
                if digest != excluded[identity]['image_sha256']:
                    raise ValueError(f'Excluded image changed: {identity}')
                verified_exclusions.add(identity)
                continue
            if digest in hashes:
                raise ValueError(f'Duplicate decoded image: {image} and {hashes[digest]}. Resolve before splitting.')
            hashes[digest] = str(image)
            records.append({'id': f'{category}/{image.name}', 'category': category,
                            'image': image.relative_to(root).as_posix(),
                            'masks': [p.relative_to(root).as_posix() for p in masks],
                            'image_sha256': digest,
                            'mask_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in masks]})
        orphan = set(folder.glob('*_mask*.png')) - used_masks
        if orphan:
            raise ValueError(f'Orphan masks in {folder}: {sorted(p.name for p in orphan)}')
    if verified_exclusions != set(excluded):
        raise ValueError('An exclusion refers to an image absent from this dataset')
    return records


def make_splits(records, seed=42):
    rng = random.Random(seed)
    splits = {name: [] for name in ('train', 'val', 'test')}
    for category in CATEGORIES:
        group = sorted((r for r in records if r['category'] == category), key=lambda r: r['id'])
        rng.shuffle(group)
        n_test = max(1, round(len(group) * .15))
        n_val = max(1, round(len(group) * .15))
        splits['test'].extend(group[:n_test])
        splits['val'].extend(group[n_test:n_test+n_val])
        splits['train'].extend(group[n_test+n_val:])
    return {'protocol': 'stratified-image-level-70-15-15-v1', 'seed': seed, 'splits': splits}


def load_sample(root, record, size=128):
    root = Path(root)
    with Image.open(root / record['image']) as source:
        shape = source.size
        image = np.asarray(source.convert('L').resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    union = np.zeros((shape[1], shape[0]), dtype=bool)
    for path in record['masks']:
        with Image.open(root / path) as source:
            if source.size != shape:
                raise ValueError(f'Mask/image dimension mismatch: {path}')
            union |= np.asarray(source.convert('L')) > 127
    if record['category'] == 'normal' and union.any():
        raise ValueError(f'Normal image has foreground annotation: {record["id"]}')
    mask = np.asarray(Image.fromarray(union.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)) > 127
    return image[..., None], mask.astype(np.float32)[..., None]


def prepare(root, destination, exclusions=None):
    destination = Path(destination)
    excluded = json.loads(Path(exclusions).read_text()) if exclusions else []
    manifest = make_splits(discover(root, excluded))
    manifest['exclusions'] = excluded
    # Validate every mask before accepting a manifest.
    for records in manifest['splits'].values():
        for record in records:
            load_sample(root, record)
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise ValueError('Existing manifest differs. Use a new path after investigating the dataset change.')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
