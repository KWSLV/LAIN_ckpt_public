# UO_lain5_object15_dynamic_retry20_clean_continuous

Clean W&B mirror and accepted-checkpoint snapshot for the staged LAIN + Object
training run.

- Snapshot: accepted logical epochs 1–10
- Source checkpoint for the resumed phase: `ckpt_23370_05.pt`
- Object forward: bypassed for epochs 1–5; enabled from epoch 6
- Text adapter: off
- Scene gate: off
- Latest accepted checkpoint: `ckpt_58425_10.pt`
- Best Unseen checkpoint: logical epoch 7, Unseen mAP `36.629509`
- Clean W&B run: [dr1ifjpc](https://wandb.ai/saaatkj-null/LAIN/runs/dr1ifjpc)

The training process was still running when this Epoch 10 snapshot was created.
Rejected rollback attempts are retained in the retry/failure records but are not
included as accepted epochs in the clean W&B curve.

Checkpoint weights are stored as GitHub Release assets; see
[CHECKPOINTS.md](CHECKPOINTS.md).

