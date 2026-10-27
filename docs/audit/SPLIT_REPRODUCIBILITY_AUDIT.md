# Split Reproducibility Audit

## Verdict

**PASS.**

The four manifests contain explicit window indices, record counts, condition
counts, class counts, and SHA-256 hashes. The manifests are tracked in
`results/splits/`.

| Fold | Train conditions | Validation | Test | Train bearing codes | Validation bearing codes | Test bearing codes | Train/validation/test windows | Manifest hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [2, 3] | 1 | 0 | K001, K002, K003, K004, KA01, KA04, KA08, KB23, KB24, KI01, KI04, KI08 | K001, K002, K003, K004, KA01, KA04, KA08, KB23, KB24, KI01, KI04, KI08 | K005, KA15, KB27, KI14 | 1488/714/248 | `67343d8524d7` |
| 1 | [0, 3] | 2 | 1 | K001, K002, K003, K004, KA01, KA04, KA08, KB23, KB24, KI01, KI04, KI08 | K001, K002, K003, K004, KA01, KA04, KA08, KB23, KB24, KI01, KI04, KI08 | K005, KA15, KB27, KI14 | 1489/744/248 | `96e4b978a2f8` |
| 2 | [0, 1] | 3 | 2 | K001, K002, K003, K004, KA01, KA04, KA08, KB23, KB24, KI01, KI04, KI08 | K001, K002, K003, K004, KA01, KA04, KA08, KB23, KB24, KI01, KI04, KI08 | K005, KA15, KB27, KI14 | 1459/744/248 | `450e3eeacd72` |
| 3 | [1, 2] | 0 | 3 | K001, K002, K003, K004, KA01, KA04, KA08, KB23, KB24, KI01, KI04, KI08 | K001, K002, K003, K004, KA01, KA04, KA08, KB23, KB24, KI01, KI04, KI08 | K005, KA15, KB27, KI14 | 1458/745/248 | `8857ac916cbf` |

## Determinism

- Held-out record IDs are fixed as `4, 8, 12, 15`.
- The test condition is the fold ID.
- Validation is chosen deterministically from the remaining condition IDs;
  there is no random split call.
- Training uses the two remaining conditions.
- All held-out bearing records are removed from both training and validation.
- Window indices use deterministic sorted file traversal and fixed starts.
- Every manifest's `indices_sha256` was recomputed from its stored indices.
- Every manifest's content hash matches the `split_hash` stored in all run rows
  for that fold.

## Reconstructing bearing identities

`data/README.md` and `data/public/paderborn/pu_4class_w8192.json` map record IDs
to bearing codes. The test codes are `K005`, `KA15`, `KI14`, and `KB27`.
Training and validation record IDs are explicitly stored in each manifest's
`record_counts`; the corresponding codes are therefore unambiguous.

## Audit issues

- None.
