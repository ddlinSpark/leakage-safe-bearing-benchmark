# SCI Manuscript Sources

The current complete manuscript is:

```text
docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx
```

It is generated from the canonical five-seed results and tracked in the final
release. The reproducible source is `src/spindle_cnn/paper/build_docx.py`.

The pre-statistics revision is retained at:

```text
docs/paper/archive/Leakage_Safe_Bearing_Benchmark_SCI_pre_statistics_revision.docx
```

Supporting records:

- `docs/reports/AUDIT_REPORT.md`: code, split, raw-source, label, and numerical audit.
- `docs/reports/REPRODUCTION_REPORT.md`: retained-result reproduction status.
- `docs/reports/EXPERIMENT_REPORT.md`: canonical and sensitivity analyses.
- `docs/reports/PAPER_REVISION_REPORT.md`: problem-evidence-change-location matrix.
- `docs/reports/PAPER_RESULT_TRACEABILITY.md`: manuscript-to-result source map.
- `docs/reports/REPRODUCIBILITY.md`: detailed end-to-end workflow.
- `docs/reports/DAMAGE_ORIGIN_ANALYSIS.md`: confounding assessment boundary.
- `docs/reports/SECOND_DATASET_ASSESSMENT.md`: external-validity assessment.
- `docs/paper/TABLE_SOURCES.md`: manuscript table-to-file provenance.
- `docs/paper/FIGURE_SOURCES.md`: figure-to-generator mapping.
- `figures/publication/figure_hashes.json`: SHA-256 and byte-size manifest for
  all 24 tracked PNG figures.
- `src/spindle_cnn/visualization/style.py`: shared 11-color journal palette and
  visual style used by every figure generator.
- `docs/reports/CONSISTENCY_REPORT.md`: automated run-count, metric, statistical, parameter,
  fold, seed, and DOCX consistency checks.
- `docs/paper/PAPER_REPRODUCIBILITY.md`: complete rebuild and audit procedure.

The earlier local file
`主轴故障诊断方法_SCI修订版.docx` is retained only as a historical source. It
must not be used for new numerical claims because parts of it predate the
five-seed canonical rerun.
