# LAIN server training code snapshot

This directory is a server-side code snapshot for the LAIN unseen-object experiments.

## Layout

- `training_scripts/`: collected training and background-launch scripts.
- `testing_scripts/`: collected evaluation, checkpoint testing, diagnostics, merging and plotting scripts.
- `docs/`: experiment ledgers, parameter records, reports, tables and figures. The server documents are overlaid with the local `G:\vision\LAIN-main\docs` records before upload.
- `scripts/`: the original runnable script tree retained so relative imports and launcher paths remain valid.
- `models/`, `datasets/`, `CLIP/`, `detr/`, `pocket/`, `utils/`: source code required by the training framework.

## Intentionally excluded

- Frozen/pretrained model files: `*.pt`, `*.pth`, `*.bin` and checkpoint directories.
- Raw HICO-DET dataset annotations/images and the server `hicodet/` data directory.
- Python bytecode/cache directories and editor metadata.
- Training outputs under `/root/autodl-tmp/Lain`; those are published separately as per-experiment GitHub Releases.

The original server files are retained after upload.
