# Release Readiness

## Decision

**NOT READY FOR v1.0.0.**

The scientific package is reproducible and numerically consistent, but public
anonymous release is blocked by identity leakage in Git metadata and retained
run configuration hostnames.

## Checks

| Requirement | Status | Evidence |
| --- | --- | --- |
| Tests pass | PASS | `33 passed in 2.12s`. |
| Working tree clean | FAIL | A release must point to a committed revision. |
| Independent numeric audit | PASS | 0 mismatches. |
| README complete | PASS | `docs/audit/REPOSITORY_AUDIT.md`. |
| CITATION present | PASS | Valid CFF 1.2.0 software record; version is 0.3.0 until the release tag is assigned. |
| License clear | PASS | MIT software scope plus explicit third-party data exclusion. |
| `.gitignore` complete | PASS | Checked against representative ignored and retained paths. |
| No secrets | PASS | No API key, token, private key, password, or credential URL detected. |
| No personal metadata | FAIL | Commit email and run-config hostname findings. |
| Raw results available | PASS | 320 canonical and 40 external rows. |
| Statistics reproducible | PASS | `results/hierarchical_statistics.csv`. |
| Figures reproducible | PASS | 12 method/data figures from tracked scripts. |
| Split manifests reproducible | PASS | Four fixed manifests with indices and hashes. |

## True release blockers

- Remove/expose-control personal commit metadata and workstation hostnames before anonymous public release.
- Commit the audited release candidate so the release points to a clean immutable revision.
- Remove redistributable-restricted raw data or model checkpoints.

## Non-blocking follow-up

- Exact transitive dependency locking remains outstanding.
- The CFF record uses the repository alias until legal author names and ORCIDs
  are confirmed after the anonymity decision.
- A GitHub tag and GitHub Release have not been created.
- No Zenodo DOI has been created.

## Severity summary

- **CRITICAL:** public identity exposure through repository ownership, Git
  author metadata, and retained workstation hostnames for anonymous review.
- **MAJOR:** the release candidate is not yet committed as a clean immutable
  revision.
- **MINOR:** no exact transitive dependency lock; 7-Zip and Pandoc are external
  tools required only for specific workflows.
- **INFORMATIONAL:** no API secrets were found; no raw data or model checkpoints
  are tracked; history contains only small prediction/result artifacts.
