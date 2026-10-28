# Release Readiness

## Decision

**NOT READY FOR v1.0.0.**

The scientific package is reproducible and numerically consistent, but a
privacy-clean public release is blocked by identity metadata in Git history and
retained run configuration hostnames.

## Checks

| Requirement | Status | Evidence |
| --- | --- | --- |
| Tests pass | PASS | `41 passed in 2.26s`. |
| Working tree clean | FAIL | A release must point to a committed revision. |
| Independent numeric audit | PASS | 0 mismatches. |
| Internal citations and PDF navigation | PASS | `docs/reports/REFERENCE_LINK_REPORT.md`. |
| README links and Python commands | PASS | `docs/reports/README_LINK_CHECK.md`. |
| README complete | PASS | `docs/audit/REPOSITORY_AUDIT.md`. |
| CITATION present | PASS | Valid CFF 1.2.0 record for Huadong Lin; version 1.0.0. |
| License clear | PASS | MIT software scope plus explicit third-party data exclusion. |
| `.gitignore` complete | PASS | Checked against representative ignored and retained paths. |
| No secrets | PASS | No API key, token, private key, password, or credential URL detected. |
| No personal metadata | FAIL | Commit email and run-config hostname findings. |
| Raw results available | PASS | 320 canonical and 40 external rows. |
| Statistics reproducible | PASS | `results/hierarchical_statistics.csv`. |
| Figures reproducible | PASS | 24 tracked PNG figures from tracked scripts. |
| Split manifests reproducible | PASS | Four fixed manifests with indices and hashes. |

## True release blockers

- Remove/expose-control personal commit metadata and workstation hostnames before privacy-clean public release.
- Commit the audited release candidate so the release points to a clean immutable revision.

## Non-blocking follow-up

- Exact transitive dependency locking remains outstanding.
- The CFF author is Huadong Lin; ORCID and release date remain unassigned.
- A GitHub tag and GitHub Release have not been created.
- No Zenodo DOI has been created.

## Severity summary

- **CRITICAL:** privacy review of public authorship, Git metadata, and retained
  workstation hostnames.
- **MAJOR:** the release candidate is not yet committed as a clean immutable revision.
- **MINOR:** no exact transitive dependency lock; 7-Zip and Pandoc are external
  tools required only for specific workflows.
- **INFORMATIONAL:** no API secrets were found; no raw data or model checkpoints
  are tracked; history contains only small prediction/result artifacts.
