# Leakage-Safe Benchmark Audit Report

## Verdict

**PASS at run level.** The canonical manifests, cache metadata, and restored source files show disjoint fitting and test windows, unseen physical bearings, and disjoint operating conditions. Re-standardizing the restored raw windows reproduces the canonical cache elementwise.

No **LEAKAGE RISK**, **PAPER-CODE MISMATCH**, or **REPRODUCTION FAILURE** was detected in the checks below.

## Evidence

- Cache: `data\public\paderborn\pu_4class_w8192.npz`
- Cache SHA-256: `a898abcb63e17ee68f2014d19b4d0f675b6c4d19a9abc89376031a78c3a23df4`
- Cache windows: 3,939
- Cached raw shape: `[3939, 3, 8192]`
- Cached CWT shape: `[3939, 3, 96, 96]`
- Window / stride: 8192 / 8192
- Source measurement files represented: 127
- Measurement mapping: **VERIFIED from original measurement filenames**
- Raw-amplitude audit cache SHA-256: `cebb85cbf15902380c9fb56920a56b8829c0babf1a2cf4e73c76929a575a8410`
- Source-file read errors declared in metadata: 1

## Audit status key

- **VERIFIED**: reproduced directly from current code, cache metadata, or manifests.
- **UNVERIFIED**: evidence needed for direct verification is absent.
- **PAPER-CODE MISMATCH**: manuscript and executable pipeline disagree.
- **LEAKAGE RISK**: fitting and held-out data overlap by index, bearing, or condition.
- **REPRODUCTION FAILURE**: an executable result cannot be reproduced from retained artifacts.

## Check summary

| check | status | evidence |
| --- | --- | --- |
| Non-overlapping windows | VERIFIED | stride alignment and disjoint half-open ranges |
| Train/test and validation/test bearing isolation | VERIFIED | recomputed from cache record identifiers |
| Operating-condition isolation | VERIFIED | recomputed from cache condition identifiers |
| Stored split manifests | VERIFIED | index sets and SHA-256 hashes |
| Raw source window mapping | VERIFIED | source filenames and elementwise standardization check |
| Label and damage metadata | VERIFIED | 16 physical-bearing manifest rows |
| Damage-origin confounding | VERIFIED | held-out faults are all real-damage bearings |
| Canonical tables, figures, and 360 result rows | VERIFIED | independent numerical audit |
| No-SE hierarchical interval consistency | VERIFIED | paired fold-seed bootstrap reproduction |

## Window overlap

- Status: **VERIFIED**
- Minimum gap between adjacent non-overlapping ranges: 0 samples
- Duplicate exact ranges: 0

Every start is aligned to the 8192-sample stride. Within each verified source measurement segment, the next window starts at or after the previous window's exclusive end.

- The only source-file read error is the known truncated `N15_M01_F10_KA08_2.mat`; the canonical build skips the same unreadable file, so the two caches retain identical row metadata.

## Bearing and condition isolation

- Overall status: **VERIFIED**

| fold | train_conditions | validation_conditions | test_conditions | train_test_bearings | validation_test_bearings | train_test_indices | validation_test_indices | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | N15_M07_F04;N15_M07_F10 | N15_M01_F10 | N09_M07_F10 |  |  | 0 | 0 | VERIFIED |
| 1 | N09_M07_F10;N15_M07_F10 | N15_M07_F04 | N15_M01_F10 |  |  | 0 | 0 | VERIFIED |
| 2 | N09_M07_F10;N15_M01_F10 | N15_M07_F10 | N15_M07_F04 |  |  | 0 | 0 | VERIFIED |
| 3 | N15_M01_F10;N15_M07_F04 | N09_M07_F10 | N15_M07_F10 |  |  | 0 | 0 | VERIFIED |

The four held-out codes are K005, KA15, KI14, and KB27. They are absent from every training and validation partition.

## Manifest reproduction

- Status: **VERIFIED**

| fold | index_sets_match | index_hashes_match | status |
| --- | --- | --- | --- |
| 0 | True | True | VERIFIED |
| 1 | True | True | VERIFIED |
| 2 | True | True | VERIFIED |
| 3 | True | True | VERIFIED |

## Damage-origin distribution

| fold | split | damage_origin | windows | bearings |
| --- | --- | --- | --- | --- |
| 0 | test | none | 62 | 1 |
| 0 | test | real | 186 | 3 |
| 0 | train | artificial | 496 | 4 |
| 0 | train | none | 496 | 4 |
| 0 | train | real | 496 | 4 |
| 0 | validation | artificial | 217 | 4 |
| 0 | validation | none | 249 | 4 |
| 0 | validation | real | 248 | 4 |
| 1 | test | none | 62 | 1 |
| 1 | test | real | 186 | 3 |
| 1 | train | artificial | 496 | 4 |
| 1 | train | none | 496 | 4 |
| 1 | train | real | 497 | 4 |
| 1 | validation | artificial | 248 | 4 |
| 1 | validation | none | 248 | 4 |
| 1 | validation | real | 248 | 4 |
| 2 | test | none | 62 | 1 |
| 2 | test | real | 186 | 3 |
| 2 | train | artificial | 465 | 4 |
| 2 | train | none | 497 | 4 |
| 2 | train | real | 497 | 4 |
| 2 | validation | artificial | 248 | 4 |
| 2 | validation | none | 248 | 4 |
| 2 | validation | real | 248 | 4 |
| 3 | test | none | 62 | 1 |
| 3 | test | real | 186 | 3 |
| 3 | train | artificial | 465 | 4 |
| 3 | train | none | 497 | 4 |
| 3 | train | real | 496 | 4 |
| 3 | validation | artificial | 248 | 4 |
| 3 | validation | none | 248 | 4 |
| 3 | validation | real | 249 | 4 |

All four held-out fault codes carry real (lifetime) damage. The fitting fault bearings mix artificial and real damage for single-location faults; both fitting multiple-damage bearings carry real damage. Damage origin therefore changes together with bearing identity in the primary holdout.

## Numerical and manuscript audit

- Status: **PASS**
- Mismatches: 0
- Canonical / external rows: 320 / 40
- Manuscript tables / figures parsed: 18 / 23

## Evidence files

- `reports/window_audit.csv`: one row per cached window with source measurement when available.
- `reports/bearing_overlap_check.csv`: bearing, condition, and index overlap by fold.
- `reports/label_audit.csv`: label and damage metadata by bearing and condition.
- `results/damage_origin_manifest.csv`: split-by-condition damage-origin manifest.
- `artifacts/local/current_numeric_audit.json`: independent numerical audit.

## Reproduction command

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.analysis.protocol_audit --run-numeric-audit
```

The reported test performance is not used to choose any audit threshold, model, preprocessing setting, or pseudo-label policy.
