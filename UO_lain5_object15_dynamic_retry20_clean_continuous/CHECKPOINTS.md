# Checkpoint downloads

The model weights are stored as GitHub Release assets because each checkpoint
exceeds GitHub's normal repository-file limit.

Release: [Epoch 10 snapshot](https://github.com/KWSLV/LAIN_ckpt_public/releases/tag/UO_lain5_object15_dynamic_retry20_clean_continuous_epoch10)

| Checkpoint | Meaning | Unseen mAP | Size | SHA-256 | Download |
|---|---|---:|---:|---|---|
| `best_unseen.pt` | Historical best, logical epoch 7 | **36.629509** | 700.68 MiB | `0a09a1dc15436838f113d9195c39a59f2c0dbff709de3a0ffcdd597eb84c2d11` | [Download](https://github.com/KWSLV/LAIN_ckpt_public/releases/download/UO_lain5_object15_dynamic_retry20_clean_continuous_epoch10/best_unseen.pt) |
| `ckpt_58425_10.pt` | Latest accepted, logical epoch 10 | 36.598580 | 700.68 MiB | `9003578c542e4944579bd8bada80978ebd1a8179b21df7c2f8cade69fc72d45f` | [Download](https://github.com/KWSLV/LAIN_ckpt_public/releases/download/UO_lain5_object15_dynamic_retry20_clean_continuous_epoch10/ckpt_58425_10.pt) |

## Accepted Unseen mAP

| Logical epoch | Object forward | Object LR | Unseen mAP |
|---:|:---:|---:|---:|
| 1 | No | 2e-4 | 31.408983 |
| 2 | No | 2e-4 | 33.072958 |
| 3 | No | 2e-4 | 34.263313 |
| 4 | No | 2e-4 | 36.241247 |
| 5 | No | 1e-4 | 36.415097 |
| 6 | Yes | 2e-4 | 36.356422 |
| 7 | Yes | 2e-4 | **36.629509** |
| 8 | Yes | 2.5e-5 | 36.491937 |
| 9 | Yes | 1.25e-5 | 36.483326 |
| 10 | Yes | 1.25e-5 | 36.598580 |

Training arguments, accepted diagnostics, retry/rollback history, failure
records, and the console-log snapshot are stored in this repository directory.

