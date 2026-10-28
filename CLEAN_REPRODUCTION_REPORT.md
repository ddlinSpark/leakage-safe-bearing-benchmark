# Clean Reproduction Report

## Verdict

The local committed source snapshot reproduces the checked training smoke run,
the retained result counts, the manuscript tables, and the final DOCX package
through an isolated Git clone.

- Isolated local Git clone: **PASS**
- Test suite inside the clone: **PASS**, 30 tests
- Canonical dataset hash: **PASS**
- Raw/Fold 0/seed 42 training smoke: **PASS**
- Manuscript numerical audit after rebuilding the DOCX: **PASS**, 0 mismatches
- Fresh clone directly from the named public GitHub remote: **UNVERIFIED**

This report does not claim that a fresh GitHub clone was tested. The current
revision is not pushed: the named remote head is
`8975858ccbf62760c20fed1eed0eb45f525e5cee`, whereas the clean source snapshot
is `1cee539bac572e50091cacf6905b43cce318fe58`. A public clone therefore
contains the earlier remote revision, not this clean snapshot.

## Provenance

| Item | Value |
| --- | --- |
| Local source snapshot | `artifacts/local/clean_source_final_20261028` |
| Snapshot commit | `1cee539bac572e50091cacf6905b43cce318fe58` |
| Commit message | `Clean reproduction snapshot 2026-10-28 final` |
| Isolated clone | `artifacts/local/clean_clone_from_snapshot_20261028` |
| Clone method | `git clone --no-hardlinks` from the local snapshot |
| Snapshot Git state before clone | clean working tree |
| Tracked files | 4,553 |
| Clone Git state after dependency injection | clean |
| Recorded run `git_dirty` | `False` |

The source snapshot contains the non-ignored source, configurations, split
manifests, results, figures, tests, reports, and manuscript source builders.
The raw Paderborn archives, the 605,098,843-byte canonical NPZ, model weights,
and local rendered-page evidence are not redistributed.

This report and the Windows clone guidance were added to the author workspace
after the isolated clone test. They are documentation-only changes; the tested
code, configurations, results, figures, and manuscript content remain those of
commit `1cee539bac572e50091cacf6905b43cce318fe58`.

The canonical cache was copied into the isolated clone for the training smoke:

```text
data/public/paderborn/pu_4class_w8192.npz
SHA-256 a898abcb63e17ee68f2014d19b4d0f675b6c4d19a9abc89376031a78c3a23df4
```

The cache hash equals the run-level `dataset_hash`. The split hash was also
identical between canonical and clone smoke rows:

```text
67343d8524d7b7322383bc1e458d64e1da56fd543827d21317bfe8461f0f6beb
```

## Environment

| Component | Recorded value |
| --- | --- |
| Operating system | Windows 11 |
| Python | 3.13.5 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| cuDNN | 91900 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| PyWavelets | 1.8.0 |
| Device | NVIDIA GeForce RTX 5080 |

The clean clone used the already-installed, recorded environment. No package
versions were selected using test data.

## Clone and Windows path handling

The first local clone copied the Git objects successfully, but checkout failed
when the deeply nested workspace plus a retained self-training diagnostic
filename exceeded the default Windows `MAX_PATH` limit. The longest tracked
relative path is 153 characters. This is a filesystem-path issue, not a Git
content or numerical mismatch.

The clone was completed by enabling Git long-path support and restoring the
worktree:

```powershell
git -C artifacts/local/clean_clone_from_snapshot_20261028 `
  config core.longpaths true
git -C artifacts/local/clean_clone_from_snapshot_20261028 `
  restore --source=HEAD --worktree :/
git -C artifacts/local/clean_clone_from_snapshot_20261028 `
  reset --mixed HEAD
```

The resulting checkout contains 4,553 tracked files and reports no modified or
untracked non-ignored files. For a new Windows clone, use
`git -c core.longpaths=true clone ...` or select a short destination directory.
The same guidance is now present in `REPRODUCE.md` and
`REPRODUCIBILITY.md`.

## Executed checks

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Result: `30 passed in 2.09s`.

The end-to-end training smoke used the primary configuration, one fold, one
seed, and no dirty-worktree override:

```powershell
python -m spindle_cnn.experiments.run_benchmark `
  --config configs/paderborn_main.json `
  --experiment main `
  --output-dir artifacts/clone_smoke `
  --seeds 42 `
  --folds 0 `
  --methods raw_cnn
```

The run loaded 3,939 windows, recorded the clean clone commit, and completed at
test Macro-F1 `0.3205`.

## Training reproduction

The canonical and clean-clone rows were matched by `raw_cnn`, fold `0`, and seed
`42`. All 25 selected numeric fields agree within machine precision. The
maximum absolute difference is `8.32667268468867e-17`; three fields differ only
in the final decimal representation written to CSV:

| Field | Canonical | Clean clone | Absolute difference |
| --- | ---: | ---: | ---: |
| `raw_ece` | 0.2189397294325984 | 0.21893972943259848 | 8.33e-17 |
| `raw_brier` | 0.1936776203409335 | 0.19367762034093358 | 8.33e-17 |
| `best_val_loss` | 0.0854179337620735 | 0.08541793376207352 | 1.39e-17 |

Prediction and optimization results:

| Quantity | Canonical | Clean clone |
| --- | ---: | ---: |
| Test Macro-F1 | 0.3205054825882636 | 0.3205054825882636 |
| Test accuracy | 0.4717741935483871 | 0.4717741935483871 |
| Macro precision | 0.2433206106870229 | 0.2433206106870229 |
| Macro recall | 0.4717741935483871 | 0.4717741935483871 |
| Calibrated ECE | 0.5087874763865246 | 0.5087874763865246 |
| Calibrated NLL | 18.054552566667947 | 18.054552566667947 |
| Brier score | 0.2567518099883856 | 0.2567518099883856 |
| Best validation Macro-F1 | 1.0 | 1.0 |
| Selected epoch | 6 | 6 |
| Epochs executed | 10 | 10 |
| Parameters | 244,740 | 244,740 |
| GFLOPs/sample | 0.676857856 | 0.676857856 |

The run identifier and configuration hash differ because the configuration
provenance records the clean clone commit `1cee...` instead of the historical
canonical execution commit `470d...`. The dataset hash, split hash, model
configuration, selected epoch, and resulting predictions do not differ.

Hardware-sensitive values are not treated as equality checks:

| Timing | Canonical | Clean clone |
| --- | ---: | ---: |
| Training time (s) | 4.7310 | 4.8620 |
| Inference latency (ms) | 26.6524 | 27.4547 |

These small timing changes are expected from process and GPU scheduling and do
not enter a numerical reproduction pass/fail decision.

## Manuscript rebuild and audit

The final DOCX is intentionally ignored as a local paper binary. The first
clean numerical verification therefore reported one availability mismatch:
the cloned source did not initially contain the generated manuscript. This was
not a table or result mismatch.

The manuscript was rebuilt inside the clean clone from its committed results
and publication figures:

```powershell
python -m spindle_cnn.paper.build_docx `
  --results-dir results `
  --figures-dir figures/publication `
  --output docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx
python scripts/verify_manuscript_numbers.py
```

Result:

```text
PASS: paper_metrics.json matches canonical result tables and the
independent DOCX/table audit reports zero mismatches.
```

The canonical and clean rebuilt DOCX packages have the same 6,561,661-byte
size, the same 41 ZIP entries, and byte-identical content for all 41 entries.
Whole-file SHA-256 values differ because ZIP container metadata, including
entry timestamps, is not part of the document content:

| Package | SHA-256 |
| --- | --- |
| Canonical final DOCX | `5e0c69fb6ef555f17b74dd0adea670752153178fc2137906ba4f397736be4605` |
| Clean-clone rebuild | `ae359f43de6b45adac52bf0d35c90198bc816f79ce30ab798ba1c0be32e4a93b` |

The package-level equality establishes that the main workspace's 30-page
visual and accessibility review applies to the same document content rebuilt
in the clone.

`scripts/audit_final_manuscript.py` also expects an ignored pre-revision backup
and author-side rendered PNG pages. A minimal clone does not contain those
local archival files, so that specific audit command raises a missing-backup
error there. This is an evidence-packaging limitation, not a manuscript
mismatch. The independent numerical DOCX audit succeeds in the clone, and the
full final manuscript audit succeeds in the author workspace where the render
evidence is retained.

## Scope and remaining limitations

The clean-room work did not rerun all 820 retained training rows. It reran the
complete test suite, a clean end-to-end training smoke, and the full independent
recomputation of stored canonical and external predictions and manuscript
tables. The retained canonical registry contains:

- 320 canonical evaluations;
- 40 external adaptation evaluations;
- 460 retained sensitivity training evaluations.

The clean clone is therefore a source, dependency, training-path, and numerical
integrity check. It is not a full independent repeat of every stochastic
sensitivity experiment.

## Release blockers

The remaining publication/release blockers are external to reproduction:

1. The named GitHub `main` remote does not yet contain this revision.
2. Fourteen existing Git commits have non-anonymous author metadata.
3. All 320 retained canonical run configurations contain a workstation
   hostname field.
4. The repository is public while the manuscript is labeled for double-blind
   review.

Resolving these items requires an explicit decision about pushing, making the
repository private or anonymous, and potentially rewriting Git history.
Repository history and publication visibility were not changed during this
revision.
