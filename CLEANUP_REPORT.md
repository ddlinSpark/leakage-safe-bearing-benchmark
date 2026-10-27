# Workspace Cleanup

Date: 2026-10-27

## Cleanup criterion

Files were retained only when they support one of:

- the current five-seed Paderborn benchmark;
- public-data reconstruction;
- canonical results, statistics, or publication figures;
- the current manuscript or cited reference PDFs;
- Git provenance.

## Removed

The following superseded working-tree categories were removed:

- legacy industrial raw data and derived CSV/SQL files;
- the old 1.2 GB `dataset_final/` industrial dataset;
- the 15.5 GB `outputs/` directory of old weights, caches, and exploratory runs;
- the 12.0 GB `artifacts/` directory of competition archives, extracted public
  archives, smoke runs, and document-review images;
- the unused 150 MB CWRU cache (reconstructible from the documented official
  source);
- legacy industrial preprocessing, training, calibration, prediction, and
  reporting modules;
- the duplicated pre-refactor public benchmark runners and model module;
- the outdated three-seed CSV directory under `docs/results/paderborn/`;
- old root-level figures from the industrial experiment;
- the original superseded manuscript DOCX;
- the outdated manuscript-generation package;
- Python bytecode, editable-install metadata, test caches, and firecrawl caches.

Approximately 72 GB of working-tree data and outputs were removed in total.

## Retained

- current source under `src/spindle_cnn/`;
- canonical configuration and tests;
- `data/README.md`;
- `data/public/paderborn/pu_4class_w8192.npz` and metadata;
- all 320 canonical result rows and per-run provenance under `results/`;
- 14 publication figure files under `figures/publication/`;
- current manuscript `docs/paper/主轴故障诊断方法_SCI修订版.docx`;
- local reference PDFs under `docs/references/`;
- audit and reproducibility documentation.

## Recovery

Deleted tracked files remain recoverable from the pre-cleanup Git commit
`65ea368`. Deleted ignored raw datasets, model weights, and archives are not
recoverable from Git; they were local experiment artifacts and are not part of
the public benchmark or paper evidence.
