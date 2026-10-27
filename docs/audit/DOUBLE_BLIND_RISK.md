# Double-Blind Risk

## Verdict

**CRITICAL risk for strict double-blind review.**

| Exposure vector | Current state | Risk |
| --- | --- | --- |
| GitHub repository URL in manuscript | not found in DOCX text | PASS |
| Public repository owner identity | Named GitHub account is visible | CRITICAL |
| CITATION author identity | Repository alias is present | CRITICAL if public during review |
| Git commit author metadata | Non-anonymous historical email metadata exists | CRITICAL |
| Current Git author configuration | Non-anonymous | CRITICAL if a release commit is created unchanged |
| README personal name | No full personal name detected | PASS |
| Run-config hostname | Workstation hostname remains in tracked run configs | CRITICAL if it reveals identity |

The manuscript avoids including the public GitHub URL, but the repository owner,
citation metadata, and commit history still connect the archive to a named
account. This is incompatible with strict anonymous review unless the journal
allows a de-anonymized code link or the authors use an approved anonymous
artifact service.

## Appropriate options

- Keep the named repository private until review completes and provide an
  anonymous source archive through the journal system.
- Create a separate anonymized mirror with neutral repository metadata,
  anonymized commit authors, and hostname/path-scrubbed history.
- Do not claim anonymization while leaving the named GitHub link, CFF owner, or
  historical commit email metadata publicly visible.

No repository visibility change or history rewrite has been performed.
