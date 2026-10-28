# Zenodo Preparation

## Status

Preparation only. No GitHub Release, tag, or Zenodo DOI has been created.

## Proposed archive metadata

- Version: `1.0.0`
- Release title: `Leakage-Safe Bearing Fault Diagnosis Benchmark v1.0.0`
- Resource type: Software
- License: MIT
- Repository: `https://github.com/ddlinSpark/leakage-safe-bearing-benchmark`
- Related manuscript: `Leakage-Safe Cross-Domain Benchmarking for Bearing Fault
  Diagnosis: Time-Frequency Fusion, Calibration, and Self-Training`
- Creator: Huadong Lin
- DOI: leave unassigned until Zenodo mints the archive DOI

## Proposed release notes

```text
Initial paper release.

- Leakage-safe Paderborn benchmark with bearing-identity and
  operating-condition holdout.
- 320 canonical evaluations and 40 same-protocol external-adaptation runs.
- DRSN and TFN adaptations, component ablation, label-efficiency and
  threshold-sensitivity studies.
- Calibration, pseudo-label, class-confusion, computational-cost, and
  validation-to-test gap analyses.
- Fold-aware hierarchical statistics and reproducible manuscript tables and
  figures.
```

## Procedure after all blockers are cleared

1. Resolve the named-public and conditional double-blind release decision.
2. Sanitize current and historical private metadata if a public named archive is
   approved.
3. Commit the release candidate and confirm a clean worktree.
4. Run the full tests and all release audits.
5. Update `CITATION.cff` to version 1.0.0 and add the release date.
6. Create an annotated Git tag and GitHub Release from the audited commit.
7. Enable the GitHub-Zenodo integration and archive that exact release.
8. Add the minted DOI to `CITATION.cff`, `README.md`, and the manuscript's
   data/code availability statement.

Ignored manuscript binaries and local artifacts are not automatically part of
a GitHub/Zenodo source archive. The tracked manuscript builder and result files
are sufficient to regenerate them after installing Pandoc and the renderer.
