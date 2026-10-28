# Reproduce the Paper from Source

The complete 2026 revision workflow, including raw-amplitude verification and
all sensitivity analyses, is maintained in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). This file retains the canonical
five-seed benchmark commands.

This guide starts from a new clone and does not require pretrained model
weights. The canonical reported latency values come from an NVIDIA GeForce RTX
5080 environment; use them only for software-level comparison.
Run the commands in one PowerShell session so the `$repro`, `$smoke`, and
figure variables remain defined. Reproduction outputs are written under the
ignored `artifacts/local/` directory.

## 1. Clone the repository

```powershell
git -c core.longpaths=true clone `
  https://github.com/ddlinSpark/leakage-safe-bearing-benchmark.git
Set-Location leakage-safe-bearing-benchmark
```

On Windows, `core.longpaths=true` is recommended because retained per-run
diagnostic files can exceed the default `MAX_PATH` limit when the clone is
stored in a deeply nested directory. Cloning into a short path is an
alternative. Linux and macOS normally use a sufficiently large path limit.

For anonymous review, use the anonymous repository mirror supplied by the
authors instead of this named public URL.

## 2. Create the environment

Python 3.10 or newer is supported. The recorded canonical environment used
Python 3.13.5, PyTorch 2.11.0 with CUDA 13.0, NumPy 2.1.3, pandas 2.2.3, and
an NVIDIA GeForce RTX 5080.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. Install dependencies

Install a PyTorch build compatible with the local CUDA or CPU environment, then
install the repository:

```powershell
python -m pip install torch torchvision
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
```

`requirements.txt` lists the remaining Python dependencies. `environment.yml`
is an optional Conda wrapper and defers Python package versions to
`requirements.txt`; it is not a separate canonical environment.

## 4. Download the Paderborn dataset

The repository does not redistribute raw signals or the processed cache.
Review the provider's terms before downloading.

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.public_benchmark.prepare_paderborn `
  --archive-dir data/public/paderborn/rars `
  --extract-dir data/public/paderborn/extracted `
  --seven-zip "C:\path\to\7z.exe"
```

Set `SEVEN_ZIP` instead of passing `--seven-zip`, or ensure `7z`, `7za`, or
`7z.exe` is on `PATH`. If the archives are already present, pass
`--skip-download`.

## 5. Configure the dataset path

The canonical configuration points to:

```text
data/public/paderborn/pu_4class_w8192.npz
```

If that is not the desired location, edit `configs/paderborn_main.json` and
`configs/paderborn_sota.json` so both use the same cache. The cache path is
relative to the repository root unless `BEARING_BENCHMARK_ROOT` overrides the
project root.

## 6. Build the processed cache

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.public_benchmark.pu_data `
  --data-root data/public/paderborn/extracted `
  --output data/public/paderborn/pu_4class_w8192.npz `
  --window-size 8192 `
  --stride 8192 `
  --scales 48 `
  --image-size 96 `
  --max-measurements 2
```

The command also writes `pu_4class_w8192.json`, a small metadata file retained
in Git so the manuscript builder can verify the expected tensor shape without
downloading the raw data.

## 7. Generate and verify the split manifests

The first benchmark run deterministically writes
`results/splits/fold0.json` through `fold3.json`. The split construction uses
sorted condition IDs and fixed held-out bearing record IDs; it does not use
random assignment.

Verify the committed manifests with:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q `
  tests/test_canonical_results.py::test_split_manifests_exclude_heldout_bearings_from_fitting `
  tests/test_splits.py
```

## 8. Run one Raw 1D-CNN experiment

This trains one seed on one fold and is the cheapest end-to-end check:

```powershell
$env:PYTHONPATH = "src"
$smoke = "artifacts/local/smoke-main"
New-Item -ItemType Directory -Force -Path $smoke | Out-Null
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_main.json `
  --experiment main `
  --methods raw_cnn `
  --folds 0 `
  --seeds 42 `
  --device cuda `
  --output-dir $smoke `
  --no-resume
```

Use `--device cpu` if no compatible CUDA device is available. The canonical
runner requires a clean worktree; do not use `--allow-dirty` for reported runs.
The dedicated output directory prevents a smoke run from mixing with the
published canonical CSV.

## 9. Run the canonical benchmark

This runs the main comparison, label-efficiency experiments, and threshold sweep:

```powershell
$env:PYTHONPATH = "src"
$repro = "artifacts/local/reproduction/results"
New-Item -ItemType Directory -Force -Path $repro | Out-Null
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_main.json `
  --experiment all `
  --device cuda `
  --output-dir $repro `
  --no-resume
```

The canonical result is 320 successful rows:

```text
140 main + 80 label-efficiency + 100 threshold
```

To run only one experiment family, replace `all` with `main`,
`label_efficiency`, or `threshold`.

## 10. Run the label-efficiency experiments

This optional family-only command uses its own output directory so it does not
duplicate rows after Step 9:

```powershell
$env:PYTHONPATH = "src"
$labelOnly = "artifacts/local/reproduction/label-only"
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_main.json `
  --experiment label_efficiency `
  --device cuda `
  --output-dir $labelOnly `
  --no-resume
```

## 11. Run the threshold-sensitivity experiments

This optional family-only command also uses a separate output directory:

```powershell
$env:PYTHONPATH = "src"
$thresholdOnly = "artifacts/local/reproduction/threshold-only"
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_main.json `
  --experiment threshold `
  --device cuda `
  --output-dir $thresholdOnly `
  --no-resume
```

## 12. Run the component ablation

The ablation table uses the matched main rows. To materialize those rows with
the ablation wrapper:

```powershell
$env:PYTHONPATH = "src"
$ablationOnly = "artifacts/local/reproduction/ablation-only"
python -m spindle_cnn.experiments.run_ablation `
  --config configs/paderborn_main.json `
  --device cuda `
  --output-dir $ablationOnly `
  --no-resume
```

## 13. Run external DRSN adaptation

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_sota.json `
  --experiment sota `
  --methods drsn `
  --device cuda `
  --output-dir "$repro/sota" `
  --no-resume
```

## 14. Run external TFN adaptation

The historical machine-readable key remains `tfnet`; the paper display name is
TFN.

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_sota.json `
  --experiment sota `
  --methods tfnet `
  --device cuda `
  --output-dir "$repro/sota" `
  --no-resume
```

The two external adaptations produce 20 runs each and 40 rows in total.

## 15. Generate descriptive summaries and paired statistics

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.analysis.statistics `
  --results "$repro/raw_runs.csv" `
  --output-dir $repro
python -m spindle_cnn.analysis.report --results-dir $repro
python -m spindle_cnn.analysis.sota `
  --results-dir $repro `
  --external-dir "$repro/sota"
```

## 16. Generate fold-aware hierarchical statistics

This is the primary paired-inference analysis. It resamples fold first and seed
second and retains all 20 fold-seed pairs.

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.analysis.hierarchical `
  --results-dir $repro `
  --external-dir "$repro/sota" `
  --output "$repro/hierarchical_statistics.csv" `
  --bootstrap-samples 20000
```

The exact four-fold sign-flip test is reported as a coarse sensitivity check;
its minimum attainable nonzero two-sided p-value is 0.125.

## 17. Recompute metrics from saved predictions

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.analysis.recheck_run `
  --results-dir $repro --all
python -m spindle_cnn.analysis.recheck_run `
  --results-dir "$repro/sota" --all
```

The expected reports are `results/rechecks/all_runs.json` and
`results/sota/rechecks/all_runs.json`.

## 18. Generate figures

```powershell
$env:PYTHONPATH = "src"
$figures = "artifacts/local/reproduction/figures"
python -m spindle_cnn.experiments.make_method_figures `
  --output-dir $figures
python -m spindle_cnn.experiments.make_figures `
  --results-dir $repro `
  --output-dir $figures
```

Methodology diagrams are generated from code and configuration. All Figure 4
through Figure 12 panels are generated from result CSV or per-run diagnostics.

## 19. Generate tables and the manuscript

The DOCX builder loads result values from CSV and generates Tables 1-13:

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.paper.build_docx `
  --results-dir $repro `
  --figures-dir "artifacts/local/reproduction/figures" `
  --output "artifacts/local/reproduction/Leakage_Safe_Bearing_Benchmark_SCI_final.docx"
```

Native Word equations require Pandoc. Install it or set `PANDOC_EXE` to the
executable path.

## 20. Verify manuscript numbers

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_submission.py `
  --results-dir $repro `
  --paper artifacts/local/reproduction/Leakage_Safe_Bearing_Benchmark_SCI_final.docx `
  --json-output artifacts/local/reproduction/pre_submission_audit.json
python scripts/audit_final_manuscript.py
python -m pytest -q
```

A successful verification reports zero numerical mismatches and all tests
passing. `audit_final_manuscript.py` intentionally audits the canonical tracked
release manuscript; `docs/audit/PAPER_RESULT_CONSISTENCY.md` records the
current result.

## Optional Release audit

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_release_repository.py
```

This regenerates the repository, release-readiness, privacy, and provenance
audits without changing experiments or creating a GitHub Release.
