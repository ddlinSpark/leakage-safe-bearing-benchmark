# Canonical Results

This directory declares the formal paper result contract. The large row-level
matrices and per-run archives remain in their verified compatibility locations
under `results/`; `manifest.json` records their paths, sizes, row counts, and
SHA-256 values.

The primary row matrix is:

```text
results/raw_runs.csv
```

It contains 320 successful canonical evaluations:

- 140 main;
- 80 label-efficiency;
- 100 threshold-sensitivity.

The separate external adaptation block is:

```text
results/sota/raw_runs.csv
```

It contains 40 successful same-protocol DRSN/TFN adaptations.

Do not edit these files manually or substitute a sensitivity result for a
canonical row. Refresh the index with:

```powershell
python scripts/build_results_catalog.py
```
