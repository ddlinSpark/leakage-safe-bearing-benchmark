# Git History Privacy Audit

## Verdict

**FAIL for privacy-clean public release.**

The scan covered all reachable Git history and the current tracked text. No API
key, access token, private key, password, credential URL, or private server IP
was detected. Personal metadata remains and is summarized without reproducing
the private values.

| Finding | Count / result | Risk |
| --- | ---: | --- |
| Reachable commits | 27 | informational |
| Distinct commit author email addresses | 2 | informational |
| Commits with non-anonymous author email metadata | 14 | privacy review required |
| Current Git author setting will create anonymous commits | NO | privacy review required |
| Current canonical run configs with hostname field | 320 of 320 | MAJOR privacy |
| API keys / tokens / private keys / passwords | 0 detected | PASS |
| Private IP or credential-bearing URLs | 0 detected | PASS |
| Raw dataset archives/caches in reachable history | 4 | FAIL |
| Model checkpoints in reachable history | 0 | PASS |
| Small saved-logit `.npz` diagnostics in history | 822 | INFORMATIONAL |
| Largest reachable blob | 6.37 MiB | INFORMATIONAL |

## Historical match locations

Values are intentionally omitted.

| Risk type | First commit | First file / location |
| --- | --- | --- |
| `email_literal` | `5c16382a1fab` | `REPRODUCIBILITY.md` |
| `absolute_windows_path` | `9006d5608577` | `REPRODUCIBILITY.md`; 5 additional match locations |
| `hostname_field` | `673bb999a646` | `src/spindle_cnn/experiments/common.py`; 642 additional match locations |

## Current candidate-tree matches

| Risk type | First file / location |
| --- | --- |
| `email_literal` | `REPRODUCIBILITY.md` |
| `absolute_windows_path` | `REPRODUCIBILITY.md` |
| `hostname_field` | `results/runs/label_efficiency__f0__s123__cwt_gru__l0.1000__4fccc07576/config.json`; 319 additional files |

The current source tree no longer contains platform-specific executable
paths. Historical absolute-path matches remain a portability concern but did
not expose a user-home directory or private server address in the scanned
history.

A normal follow-up commit cannot remove metadata from existing public history.
Rewriting history requires explicit author approval, a complete backup, and a
coordinated force-with-lease update.

## Prepared git filter-repo procedure, not executed

```powershell
git clone --mirror https://github.com/ddlinSpark/leakage-safe-bearing-benchmark.git repository-mirror.git
git clone --mirror repository-mirror.git repository-backup.git
cd repository-mirror.git
git filter-repo --mailmap ../private-mailmap.txt --replace-text ../private-replacements.txt
git push --force-with-lease
```

`private-mailmap.txt` and `private-replacements.txt` must be created locally,
must contain the actual sensitive values, and must never be committed. The
replacement file can map literal hostnames and personal paths to neutral labels.
Do not execute this procedure until the anonymity and repository-visibility
decision is made.
