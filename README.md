# Leakage-Safe Bearing Fault Diagnosis Benchmark

Official implementation and reproducibility package for:

**Leakage-Safe Cross-Domain Benchmarking for Bearing Fault Diagnosis: Time-Frequency Fusion, Calibration, and Self-Training**

This repository evaluates whether commonly used bearing-fault-diagnosis
techniques remain effective when both **physical bearing identity** and
**operating condition** are unseen during testing.

[![Python](https://img.shields.io/badge/Python-3.10%2B-7B95C6?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-F47254?style=flat-square)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-67A583?style=flat-square)](LICENSE)
[![Protocol](https://img.shields.io/badge/Protocol-Leakage--Safe-49C2D9?style=flat-square)](docs/PROTOCOL.md)
[![Reproducible](https://img.shields.io/badge/Reproducible-Clean%20Clone-A2C986?style=flat-square)](CLEAN_REPRODUCTION_REPORT.md)
[![Paper](https://img.shields.io/badge/Paper-PDF-F59C7C?style=flat-square)](manuscript/manuscript.pdf)

## Overview

The project is a benchmark, evaluation-methodology, and negative-results study, not a new high-scoring architecture.

The benchmark enforces:

- physical-bearing-disjoint training and testing;
- operating-condition-disjoint training and testing;
- non-overlapping 8192-sample signal windows;
- validation-only early stopping and calibration;
- matched fold-seed comparisons;
- fold-first hierarchical statistical inference.

The central question is whether an advantage survives simultaneous removal of familiar bearings and conditions.

## Key Findings

Values are mean +/- sample standard deviation over four folds and five seeds.

| Method | Test Macro-F1 | Accuracy |
| --- | ---: | ---: |
| Raw 1D-CNN | 0.289 +/- 0.104 | 0.392 +/- 0.094 |
| GRU only | 0.221 +/- 0.030 | 0.242 +/- 0.025 |
| CWT 2D-CNN | 0.291 +/- 0.086 | 0.399 +/- 0.081 |
| CWT-GRU without SE | 0.307 +/- 0.103 | 0.408 +/- 0.097 |
| CWT-GRU | 0.280 +/- 0.107 | 0.383 +/- 0.092 |
| CWT-GRU + temperature scaling + self-training | 0.290 +/- 0.116 | 0.387 +/- 0.099 |

The evaluated magnitude-CWT, recurrent-fusion, SE, calibration, and
self-training configurations do not establish a stable held-out advantage over
the simple waveform baseline under this protocol.

## Generalization Gap

For the four principal models:

- validation Macro-F1 is approximately **0.797-0.853**;
- held-out test Macro-F1 is approximately **0.280-0.307**.

Strong development-domain performance therefore does not imply transfer to a
new physical bearing and operating condition. The held-out domain also differs
in damage origin for some bearings, so these factors cannot be causally
separated by the primary design.

The study reports "no stable advantage was established" or "the evaluated
configuration did not demonstrate a repeatable improvement." These are not
claims that a representation or architecture fails universally.

## Experiment Status

> **All reported experiments completed as designed; not all evaluated methods produced positive or statistically stable gains.**

"Completed" refers to successful execution and reproducibility, not positive
performance.

| Experiment group | Status | Main outcome |
| --- | --- | --- |
| Main benchmark | Completed | No stable advantage established for the evaluated CWT/recurrent/SE variants |
| Calibration | Completed | Improved probability metrics for selected CWT models but did not change argmax classification |
| Self-training | Completed | No stable diagnostic improvement established |
| Threshold sensitivity | Completed | Higher pseudo-label accuracy did not guarantee higher held-out Macro-F1 |
| Label efficiency | Completed | Performance depended strongly on the verified-label budget |
| External baselines | Completed | Same-protocol adaptations did not outperform the waveform baseline |
| CWT sensitivity | Completed | Representation settings changed results but no universal CWT advantage emerged |
| Training opportunity | Completed | Additional training did not materially change the main conclusion |
| Embedding diagnostics | Completed | Post-hoc diagnostic analysis only |

## Leakage-Safe Protocol

The four test domains use operating conditions `C0-C3`. In every fold, one
condition is held out for test, a second condition is validation, and the
remaining two conditions are training. The same 12 development bearings are
used for fitting; `K005`, `KA15`, `KI14`, and `KB27` are excluded from all
training and validation data.

The canonical repetition design is:

```text
4 operating-condition folds
5 seeds: 42, 123, 2024, 2025, 3407
20 matched evaluations per method condition
```

Validation data can be used for early stopping and temperature fitting. Test
data never enters training, calibration fitting, pseudo-label generation,
threshold selection, hyperparameter selection, or model selection.

Primary paired inference resamples the four folds first and the five seeds
within each selected fold. The 20 fold-seed evaluations are not treated as 20
independent domains. The exact fold-level sign-flip test has a minimum
attainable non-zero two-sided p-value of `0.125`.

See [docs/PROTOCOL.md](docs/PROTOCOL.md) for the frozen fold table and complete
leakage boundaries.

## Main Results

The 320 canonical evaluations comprise 140 main, 80 label-efficiency, and 100
threshold-sensitivity rows. A separate block contains 40 DRSN/TFN
same-protocol adaptations.

Coverage-controlled self-training at the 50% budget changes Macro-F1 by
`+0.034` with interval `[-0.020, +0.096]`, despite offline pseudo-label
accuracy near `0.987`. High precision therefore does not establish a stable
end-task gain. Paired intervals, label-granularity results, calibration, and
all source files are detailed in [docs/RESULTS.md](docs/RESULTS.md).

## Repository Structure

```text
.
├── .github/       # CI verification
├── configs/       # Frozen experiment configurations
├── data/          # Data instructions and archive manifests; no raw signals
├── docs/          # Dataset, protocol, experiments, results, and reproduction
├── figures/       # 24 publication PNG/PDF pairs and hash manifest
├── manuscript/    # Submission manuscript and supplementary PDFs
├── results/       # Canonical, external, sensitivity, and post-hoc evidence
├── scripts/       # Stable entry points and audits
├── splits/        # Index of the frozen fold manifests
├── src/           # Models, training, evaluation, statistics, visualization
└── tests/         # Split, model, metric, result, and sensitivity checks
```

The compatibility package name remains `spindle_cnn`. The shared figure style
is `src/spindle_cnn/visualization/style.py`; `plotting.style` remains a
backward-compatible alias.

## Quick Start

Install a PyTorch build compatible with the local CUDA or CPU environment:

```bash
python -m pip install -e ".[dev]"
python scripts/smoke_test.py
```

Prepare the official Paderborn data:

```bash
python scripts/prepare_data.py
```

Run one complete fold-seed training smoke:

```bash
python scripts/run_main_benchmark.py \
  --experiment main \
  --folds 0 \
  --seeds 42 \
  --methods raw_cnn \
  --output-dir artifacts/local/smoke-main
```

Regenerate statistics, figures, and manuscript from the committed canonical
results:

```bash
python scripts/generate_statistics.py
python scripts/generate_figures.py
python scripts/build_paper_metrics.py
python -m spindle_cnn.paper.build_docx \
  --results-dir results \
  --figures-dir figures/publication \
  --output docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx
```

Run `python scripts/reproduce_all.py` for the orchestrated check. Detailed
commands and expected outputs are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Data Availability

The original Paderborn data are not redistributed. Users should obtain the
dataset from the official source and follow
[docs/DATASET.md](docs/DATASET.md).

Large raw signals, generated caches, model checkpoints, and temporary logs are
intentionally excluded from Git. Official archive names, sizes, and SHA-256
values are tracked in
[`data/manifests/paderborn_archives_sha256.csv`](data/manifests/paderborn_archives_sha256.csv).

## Reproducibility

The repository includes:

- fixed split manifests with hashes;
- canonical configurations and explicit seed definitions;
- a per-run result registry and compact prediction logs;
- fold-aware statistical scripts;
- scripts for all manuscript and diagnostic figures;
- same-protocol external adaptations;
- sensitivity and post-hoc outputs;
- automated manuscript, CSV, figure, and smoke-test checks.

All manuscript values are traceable to committed result files through
`results/paper_metrics.json`, `docs/paper/TABLE_SOURCES.md`, and
`docs/paper/FIGURE_SOURCES.md`.

Key evidence:

- [CONSISTENCY_REPORT.md](CONSISTENCY_REPORT.md)
- [CLEAN_REPRODUCTION_REPORT.md](CLEAN_REPRODUCTION_REPORT.md)
- [results/canonical/manifest.json](results/canonical/manifest.json)
- [splits/manifest.json](splits/manifest.json)
- [figures/publication/figure_hashes.json](figures/publication/figure_hashes.json)
- [FINAL_RELEASE_REPORT.md](FINAL_RELEASE_REPORT.md)

## Scope and Limitations

The benchmark is evaluated on the Paderborn dataset under one simultaneous
bearing-identity and operating-condition holdout protocol.

The held-out domain also differs in damage origin for some fault bearings.
Bearing identity, operating condition, and damage-origin effects therefore
cannot be causally separated.

The conclusions are configuration-specific empirical evidence, not universal
statements about all bearing datasets or architectures. DRSN and TFN are
same-protocol adaptations, not reproductions of their original reported
settings.

The project has used test outcomes for descriptive analysis after primary model
selection. Its leakage claim concerns the frozen training, calibration,
pseudo-label, hyperparameter, and model-selection protocol, not a claim that
the authors were cognitively unaware of test behavior during all analyses.

## Citation and License

Citation metadata is in [CITATION.cff](CITATION.cff). The manuscript PDF and
standalone supplementary PDF are under [`manuscript/`](manuscript/). Source and
derived documentation use the [MIT License](LICENSE); original datasets remain
subject to their provider's terms.
