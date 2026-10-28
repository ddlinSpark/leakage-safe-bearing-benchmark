# Frozen Split Manifests

The authoritative fold manifests are generated and retained by the benchmark
runner at:

```text
results/splits/fold0.json
results/splits/fold1.json
results/splits/fold2.json
results/splits/fold3.json
```

This root-level directory provides a stable discovery point without duplicating
the manifests. `manifest.json` contains their SHA-256 values and explicitly
names `results/splits` as the authoritative directory.

Refresh the index with:

```powershell
python scripts/build_results_catalog.py
```
