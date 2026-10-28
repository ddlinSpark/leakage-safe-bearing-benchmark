# Repository Release Audit

## Verdict

**Reproducible source structure: PASS with noted warnings.** The repository
contains the experiment implementation, configuration, fixed manifests,
per-run results, derived statistics, publication figures, manuscript builder,
and audit commands. Public v1.0.0 release remains blocked by the privacy issues
documented in `docs/audit/GIT_HISTORY_PRIVACY_AUDIT.md`.

## Repository scale

- Candidate files: 4571
- Candidate size: 44.66 MiB
- Large files at or above 1 MB: 6
- Saved-logit prediction archives: 822
  (8.58 MiB total; diagnostic, not model weights)
- Canonical result rows: 320
- External result rows: 40
- Split manifests: 4

The exact candidate file list is in `docs/audit/REPOSITORY_FILE_INVENTORY.txt`.

## Top-level inventory

| Path | Files | Size |
| --- | ---: | ---: |
| `.gitattributes` | 1 | 261 B |
| `.github` | 1 | 1.2 KiB |
| `.gitignore` | 1 | 1.6 KiB |
| `CITATION.cff` | 1 | 1013 B |
| `FINAL_RELEASE_REPORT.md` | 1 | 9.8 KiB |
| `LICENSE` | 1 | 1.3 KiB |
| `README.md` | 1 | 10.2 KiB |
| `configs` | 5 | 5.2 KiB |
| `data` | 6 | 9.5 KiB |
| `docs` | 53 | 6.63 MiB |
| `environment.yml` | 1 | 125 B |
| `figures` | 26 | 7.57 MiB |
| `pyproject.toml` | 1 | 3.0 KiB |
| `reports` | 6 | 564.9 KiB |
| `requirements.txt` | 1 | 307 B |
| `results` | 4342 | 29.00 MiB |
| `scripts` | 40 | 302.5 KiB |
| `splits` | 2 | 1.2 KiB |
| `src` | 70 | 561.5 KiB |
| `tests` | 11 | 22.4 KiB |

## Logical source layout

```text
.
├── configs/                    canonical and external configurations
├── data/                       data documentation and small cache metadata
├── docs/                       protocol, paper, figure, and table provenance
├── figures/publication/        generated publication PNG figures
├── results/
│   ├── splits/                 fixed fold manifests with all window indices
│   ├── runs/                   320 canonical per-run diagnostic directories
│   ├── rechecks/               saved-logit metric recomputation
│   └── sota/                   40 external-adaptation runs and summaries
├── scripts/                    analysis, checking, and paper wrappers
├── src/spindle_cnn/
│   ├── models/                 registered model implementations
│   ├── datasets/               leakage-safe splits and balanced budgets
│   ├── training/               training and calibrated self-training
│   ├── evaluation/             metrics, calibration, and efficiency
│   ├── experiments/            benchmark and figure entry points
│   ├── analysis/               summaries and fold-aware statistics
│   ├── public_benchmark/       public-data preparation
│   └── paper/                  manuscript and table generation
└── tests/                      split, model, metric, and result checks
```

The requested top-level `models/`, `training/`, and `evaluation/` directories
are represented under `src/spindle_cnn/`; this packaging layout is appropriate
and does not need to be flattened.

## Required artifacts

| Artifact | Status |
| --- | --- |
| `README.md` | PASS |
| `LICENSE` | PASS |
| `CITATION.cff` | PASS |
| `.gitignore` | PASS |
| `requirements.txt` | PASS |
| `environment.yml` | PASS |
| `pyproject.toml` | PASS |
| `docs/reports/CHANGELOG.md` | PASS |
| `docs/REPRODUCIBILITY.md` | PASS |
| `docs/reports/REPRODUCE.md` | PASS |
| `docs/reports/REPRODUCIBILITY.md` | PASS |
| `docs/reports/CLEAN_REPRODUCTION_REPORT.md` | PASS |
| `docs/reports/CONSISTENCY_REPORT.md` | PASS |
| `docs/reports/REFERENCE_LINK_REPORT.md` | PASS |
| `docs/reports/README_LINK_CHECK.md` | PASS |
| `FINAL_RELEASE_REPORT.md` | PASS |

| Component | Resolved path | Status |
| --- | --- | --- |
| `configs` | `configs` | PASS |
| `data` | `data` | PASS |
| `models` | `src/spindle_cnn/models` | PASS |
| `training` | `src/spindle_cnn/training` | PASS |
| `evaluation` | `src/spindle_cnn/evaluation` | PASS |
| `experiments` | `src/spindle_cnn/experiments` | PASS |
| `scripts` | `scripts` | PASS |
| `results` | `results` | PASS |
| `figures` | `figures` | PASS |
| `docs` | `docs` | PASS |

## README audit

| Requirement | Status |
| --- | --- |
| Project title | PASS |
| Benchmark positioning | PASS |
| Leakage-safe rationale | PASS |
| Dataset | PASS |
| Key findings | PASS |
| Model findings | PASS |
| Protocol | PASS |
| Experiment status | PASS |
| Quick verification | PASS |
| Full reproduction | PASS |
| External baselines | PASS |
| Statistical analysis | PASS |
| Data availability | PASS |
| Reproducibility | PASS |
| Figures | PASS |
| Citation | PASS |
| 320 canonical runs | PASS |
| 40 external runs | PASS |
| Coupled limitations | PASS |
| Unsupported promotional claims | PASS |

`SOTA` survives only as a historical filename and result-directory label; the
README describes those rows as same-protocol external adaptations and does not
claim state-of-the-art performance.

## Dependency and portability audit

- `requirements.txt` is the canonical install specification.
- `environment.yml` creates Python 3.13 and delegates packages to
  `requirements.txt`; it is an optional wrapper, not a second competing lock.
- PyTorch is intentionally installed separately so the user can select a
  compatible CUDA or CPU wheel.
- No local editable dependency path, private wheel, or user-home path remains
  in the executable source.
- Exact transitive package versions are not fully locked. Recorded run
  configurations preserve the principal runtime versions, but a future
  lockfile would improve bit-level environment reconstruction.
- Training and inference timings are identified as hardware-dependent.
- `CITATION.cff` uses the current package version `1.0.0`. `date-released` and a
  DOI are intentionally omitted until the v1.0.0 tag and Zenodo archive exist.

## Data and license audit

- `data/README.md` identifies Paderborn and CWRU, official sources, signals,
  sampling rate, windows, stride, conditions, measurements, and commands.
- **Raw data are not redistributed.**
- No raw Paderborn or CWRU data files are tracked.
- Current raw-data archive/cache candidates: 0.
- The cache `.npz` is ignored; the small dimensionality metadata JSON is
  retained so tables can be audited without redistributing signals.
- The MIT license is present and limited to software/documentation.
- Third-party dataset terms remain separate from the code license.
- No model checkpoints are tracked.
- Current model-checkpoint candidates: 0.

## `.gitignore` coverage

| Example path | Expected | Status |
| --- | --- | --- |
| `__pycache__/x.pyc` | ignored | PASS |
| `x.pyc` | ignored | PASS |
| `.venv/x` | ignored | PASS |
| `venv/x` | ignored | PASS |
| `env/x` | ignored | PASS |
| `.ipynb_checkpoints/x` | ignored | PASS |
| `.idea/x` | ignored | PASS |
| `.vscode/x` | ignored | PASS |
| `weights.pt` | ignored | PASS |
| `weights.pth` | ignored | PASS |
| `weights.ckpt` | ignored | PASS |
| `logs/x` | ignored | PASS |
| `cache/x` | ignored | PASS |
| `wandb/x` | ignored | PASS |
| `.DS_Store` | ignored | PASS |
| `Thumbs.db` | ignored | PASS |
| `.env` | ignored | PASS |
| `secret.key` | ignored | PASS |
| `archive.rar` | ignored | PASS |
| `signal.mat` | ignored | PASS |
| `cache.npz` | ignored | PASS |
| `results/raw_runs.csv` | retained | PASS |
| `results/splits/fold0.json` | retained | PASS |
| `figures/publication/figure_01_model_comparison.png` | retained | PASS |
| `data/public/paderborn/pu_4class_w8192.json` | retained | PASS |

## TFN naming

- Manuscript, README, and report display name: `TFN`.
- External label: `TFN adaptation` or `TFN (same-protocol adaptation)`.
- Historical machine-readable key: `tfnet`, retained for result integrity.
- `TFNet` remains only in the upstream comparison-code URL and a Python
  compatibility alias; it is not used as the manuscript display name.

## External-model provenance

- DRSN is a same-protocol adaptation based on the published residual shrinkage
  concept; the repository does not claim an official reproduction.
- TFN is adapted from the authors' public comparison-model code and retains the
  historical `tfnet` result key.
- `docs/paper/SOTA_RETRAINING_REPORT.md` records paper citations, source links,
  adaptation notes, measured complexity, and fold-aware comparison results.
- MTFC-T is excluded because the inspected public repository did not contain a
  sufficient executable proposed-model implementation.
- No published result from another dataset or split is used as a
  same-protocol ranking value.

## Findings

- CRITICAL: see the Git-history privacy report and conditional double-blind
  assessment.
- MINOR: exact transitive dependency lock is absent.
- MINOR: platform-specific archive extraction requires a local 7-Zip binary.
- INFORMATIONAL: the final manuscript is tracked as DOCX; PDFs are excluded.
