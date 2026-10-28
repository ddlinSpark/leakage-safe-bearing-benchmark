# Reference and Cross-Reference Link Report

## Verdict

**PASS**

PDF links were tested in a temporary, non-tracked render by resolving every link annotation destination to a target page. Source text was extracted from each annotation rectangle.

```yaml
reference_hyperlink_status: PASS
figure_cross_reference_status: PASS
table_cross_reference_status: PASS
equation_cross_reference_status: PASS
pdf_hyperlink_preservation: PASS
pdf_bookmarks: PASS
```

## Summary

| Metric | Value |
| --- | ---: |
| Total reference entries | 22 |
| Total in-text citation occurrences | 36 |
| Unique cited references | 22 |
| Missing reference targets | 0 |
| Unused references | 0 |
| Broken DOCX hyperlinks | 0 |
| Citation ranges detected | 4 |
| DOCX bookmarks | 114 |
| DOCX internal hyperlinks | 65 |
| PDF link annotations | 65 |
| PDF pages | 30 |
| Defined Figure targets | 23 of 23 |
| Defined Table targets | 18 of 18 |
| Defined Equation targets | 6 of 6 |

## Checks

| Check | Status |
| --- | --- |
| Reference targets and bookmarks | PASS |
| Figure/table/equation targets | PASS |
| Section/Appendix targets | PASS |
| Citation links | PASS |
| DOCX anchors | PASS |
| Figure/table/section/equation links | PASS |
| PDF annotation destinations | PASS |
| PDF bookmarks/outline | PASS |
| Required click samples | PASS |

## Citation details

- Cited references: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]`
- Missing reference targets: `[]`
- Unused references: `[]`
- Missing citation links: `[]`
- Detected ranges: `['[1-3]', '[11-13]', '[14,16-18]', '[16-18]']`
- Range handling: Visible range endpoints are linked independently. Intermediate numbers inside a hyphenated range remain part of the displayed range but are not separately visible hyperlink targets.

## Cross-references

- Occurrences by kind: `{'appendix': 0, 'equation': 0, 'figure': 8, 'section': 1, 'table': 9}`
- Missing cross-reference links: `[]`
- Missing figure targets: `[]`
- Missing table targets: `[]`
- Missing equation targets: `[]`
- Missing section/appendix targets: `[]`

## PDF click tests

| Display text | Source page | Target page | Status |
| --- | ---: | ---: | --- |
| `[1]` | 2 | 29 | PASS |
| `[14]` | 2 | 30 | PASS |
| `Figure 4` | 9 | 10 | PASS |
| `Table 13` | 19 | 19 | PASS |

## PDF outline

- Outline entries: 45
- Missing required outline entries: `[]`

The PDF outline contains Abstract, all numbered top-level sections, Appendix A, Data and Code Availability, and References.
