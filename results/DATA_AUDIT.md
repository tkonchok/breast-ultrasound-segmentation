# Dataset audit

Source: Kaggle mirror `aryashah2k/breast-ultrasound-images-dataset/versions/1`.

| Category | Raw images | Excluded | Included | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| Benign | 437 | 1 | 436 | 306 | 65 | 65 |
| Malignant | 210 | 1 | 209 | 147 | 31 | 31 |
| Normal | 133 | 0 | 133 | 93 | 20 | 20 |
| Total | 780 | 2 | 778 | 546 | 116 | 116 |

All 18 secondary annotation files are included in mask unions. Input/annotation dimensions, normal-mask emptiness, pairing, orphan masks, and exact decoded-image uniqueness are checked before accepting the manifest.

One exact image duplicate pair has conflicting category labels and different masks:

- `benign/benign (433).png`
- `malignant/malignant (145).png`

Both are excluded; neither annotation is assumed to be authoritative. Exclusion occurs before splitting and does not depend on model scores. The decoded-image SHA-256 is `bea8a9762e7799e6a3db2431e0719580748c65a50aabbc0df760ca3d6bd9c812`. `configs/data_exclusions.json` checks this hash when applying the rule. Raw files are retained locally. Other duplicates without explicit audit rules cause preparation to fail.

Splits are disjoint by file identity and contain unique decoded images after exclusions. This does not establish patient disjointness or eliminate near duplicates. Filenames are not treated as patient identifiers.

The exact included records, file paths, image hashes, and annotation-file hashes are saved in `splits.json`. `data_audit_raw.json` records the raw counts, duplicate identities, and all secondary-mask filenames. Neither file includes image pixels.
