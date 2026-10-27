# Pre-submission Reproducibility Audit

Date: 2026-10-27  
Published baseline commit: `8ac2d75`  
Raw-result SHA-256: `a047b0f9530f07643ef2c5c57f395a15abe045c7e8640a10119c273e46fbeb10`

## Decision

**Scientific reproducibility: PASS.**  
**Public-release privacy and double-blind readiness: FAIL.**

The experiment matrix, statistics, manuscript tables, manuscript figures,
ablation code, and leakage controls are internally consistent. The repository
should not be treated as ready for a named public release until the personal
metadata issue described below is resolved and the anonymity choice is made.

## Repository completeness

| Required item | Present | Location or note |
| --- | --- | --- |
| README | PASS | `README.md` |
| License | PASS | `LICENSE`, MIT |
| Dependency specification | PASS | `requirements.txt` and `environment.yml` |
| Configuration | PASS | `configs/paderborn_main.json` |
| Data documentation | PASS | `data/README.md` |
| Models | PASS | `src/spindle_cnn/models/` |
| Training | PASS | `src/spindle_cnn/training/` |
| Evaluation | PASS | `src/spindle_cnn/evaluation/` |
| Experiments | PASS | `src/spindle_cnn/experiments/` |
| Scripts | PASS | `scripts/` |
| Results | PASS | `results/` |
| Figures | PASS | `figures/publication/` |

The import package retains the historical name `spindle_cnn`; the repository
and distribution name are `leakage-safe-bearing-benchmark`.

## Security and privacy scan

Tracked and current working-tree text was scanned for cloud keys, GitHub tokens,
OpenAI-style keys, Slack tokens, credential-bearing URLs, private keys, W&B
credentials, database URLs, private IP addresses, user-home paths, passwords,
and personal identifiers.

| Finding | Status |
| --- | --- |
| API keys, tokens, passwords, private keys | PASS: none found |
| Credential-bearing URLs or private server addresses | PASS: none found |
| Absolute Windows/macOS/Linux user-home paths | PASS: none found |
| `.env` files or model checkpoints | PASS: none tracked |
| Python bytecode, `__pycache__`, IDE caches | PASS: none tracked |
| Raw dataset binaries and reference PDFs | PASS: none tracked |
| Personal host metadata in run records | **FAIL** |
| Personal email metadata in Git commits | **FAIL for anonymous release** |

The failure details are intentionally reported without reproducing the private
values:

- All 320 tracked per-run configuration files contain the same environment
  hostname field. This field identifies the workstation and is present in the
  published Git history.
- Every commit in the current history carries the same personal email address
  in the author metadata. The address is not repeated here.
- The public repository is associated with a named account while the manuscript
  is labeled for double-blind review.

Editing the current files alone would not remove the information from public Git
history. A history rewrite or replacement with a sanitized root commit requires
an explicit decision and a backup before force-pushing.

## Large-file and ignore review

| Check | Result |
| --- | --- |
| Tracked files | 1,713 |
| Total tracked size | Approximately 10.5 MiB |
| Largest tracked file | Publication confusion-matrix PNG, 844 KiB |
| Raw Paderborn cache | Ignored; local only, approximately 577 MiB |
| Reference PDFs | Ignored; local only |
| Manuscript DOCX | Ignored; local deliverable |
| Tracked model checkpoints | None |
| Tracked ZIP/archive files | None |
| Tracked IDE or Python caches | None |

The local `.git` directory is approximately 1.66 GiB and contains recoverable
but non-reachable objects from the earlier project history. The current `main`
history has no reachable raw dataset blob and its largest reachable object is
below 1 MiB. No cleanup of local unreachable objects was performed during this
audit.

The `.gitignore` now covers data binaries, model weights, archives, caches,
coverage output, common editor files, OS metadata, and local secrets. Exact
dependency versions remain available only for packages recorded in each run's
environment snapshot; a complete transitive dependency lockfile is still a
reproduction WARNING.

## Numerical consistency

The independent audit script
`scripts/audit_submission.py` recomputes from `results/raw_runs.csv`:

- run counts and complete fold/seed/method/threshold keys;
- means and sample standard deviations;
- 95% t confidence intervals;
- paired differences and two-sided Wilcoxon tests;
- ablation summaries;
- threshold and label-efficiency summaries;
- parameter, FLOP, training-time, and inference-time medians;
- all 12 manuscript tables and key manuscript numeric statements.

Result:

```text
Numerical mismatches: 0
Paper tables checked: 12
Paper figures checked: 11
```

No `RESULT_MISMATCH_REPORT.md` was generated because no numerical mismatch was
found.

## Experiment completeness

All 320 canonical rows are successful, unique, and complete:

- 140 main evaluations;
- 80 saved partial-label evaluations;
- 100 threshold evaluations;
- 20 runs per main method;
- 20 runs per saved label-efficiency condition;
- 20 runs per threshold;
- five canonical seeds and four folds for every main configuration.

See `EXPERIMENT_COMPLETENESS.md` for the full table and expected missing-value
inventory.

## Leakage audit

The implemented pipeline passes the requested leakage checks for:

- non-overlapping windows;
- unseen bearing identity;
- unseen operating condition;
- disjoint train/validation/test windows;
- training-only pseudo-label candidates;
- test-label isolation from pseudo-label selection;
- validation-only early stopping;
- validation-only temperature fitting;
- per-window normalization and per-window CWT construction.

The remaining WARNING is study-level reuse of the same four test folds across
development iterations. See `LEAKAGE_AUDIT.md` for code evidence.

## Ablation verification

The ablation is implemented through real model and training paths:

- CWT-only path: `CWT2DCNN`;
- GRU-only path: `RawGRU`;
- CWT + GRU without SE: `CWTGRU(use_se=False)`;
- CWT + GRU with SE: `CWTGRU(use_se=True)` with three actual SE modules;
- calibration: the matched CWT-GRU logits transformed by validation-fitted
  temperature;
- self-training: candidate selection and retraining in `run_self_training`.

The no-SE and full models differ by 14,768 measured parameters. The temperature
row has identical logits and argmax decisions to its matched CWT-GRU run. All
main ablation conditions use the same dataset hash, split hash per fold, seeds,
label budget, preprocessing, and metrics.

## SOTA and baseline audit

Table 10 contains only models trained by this project under the same dataset,
split, windowing, preprocessing, and metric implementation. No result from
another dataset or split is used as a ranking value.

The executable registry contains only:

```text
raw_cnn, gru_only, cwt_cnn, cwt_gru_no_se,
cwt_gru, cwt_gru_temp, cwt_gru_selftrain, drsn, tfnet
```

`cnn_lstm`, `drsn_transformer`, and `mtfc_t` remain explicit, non-executable
placeholders. The repository contains no claimed reproduction of their paper
metrics. DRSN and TFNet same-protocol retraining is an added release step and
must pass its own 20-run completeness check before entering the comparison
table.

## Figure reproducibility

Regenerating both method and result figures into a separate output directory
produced byte-identical PNGs for all 11 manuscript figures.

## Corrections applied during this audit

Only non-scientific repository inconsistencies were corrected:

- updated the MIT copyright name from the old project name;
- aligned `environment.yml` with the recorded Python major/minor version;
- replaced a stale reproducibility paragraph that referenced removed code;
- expanded `.gitignore` for additional local caches, editor files, secrets,
  coverage output, and local databases;
- stopped future environment snapshots from recording the workstation hostname;
- added explicit effective self-training-round metadata to future run
  configurations;
- added the reproducible audit script and this audit documentation.

No experiment result, split, metric, threshold, model architecture, training
setting, or manuscript numerical claim was modified.

## Required actions before public submission

1. Decide whether the submission is double-blind. If it is, remove the public
   named-account repository link from the submission or make the repository
   private until acceptance.
2. Sanitize the workstation hostname from all retained run configurations.
3. Decide whether to rewrite Git author metadata. If yes, create a backup,
   rewrite history consistently, and force-push the sanitized history.
4. Re-run `scripts/audit_submission.py` and `pytest` after the history decision.
5. Consider adding an environment lockfile once the exact historical
   scikit-learn, SciPy, PyWavelets, and Matplotlib versions are recovered from
   the original environment.

## Reproduction commands

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests
python scripts/audit_submission.py --json-output artifacts/local/pre_submission_audit.json
```
