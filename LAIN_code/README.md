## [CVPR2025] Locality-Aware Zero-Shot Human-Object Interaction Detection
This repo is the official code for our CVPR2025 paper "[Locality-Aware Zero-Shot Human-Object Interaction Detection](https://arxiv.org/abs/2505.19503)".



## Installation
On the AutoDL PyTorch 2.8 / CUDA 12.8 image, install the server dependencies
from the repository root. The script keeps the image's existing PyTorch build;
it installs PyTorch 2.8 wheels only when PyTorch is absent.

```shell
cd /root/Lain/LAIN-main
bash scripts/install_server_deps.sh
```

The local `pocket` setup installs all subpackages, including `pocket.core`.


## Data preparation

### Dataset download
We provide a shell script for downloading the dataset.
You can download all the required files by running the following command:

```shell
bash scripts/download.sh
```

### Model Zoo
| Zero-shot setting |     Backbone      | Unseen | Seen  |  mAP  |
|:-----------------:|:-----------------:|:------:|:-----:|:-----:|
|       RF_UC       | ResNet50+ViT-B/16 | 32.34  | 35.17 | 34.60 | 
|       NF_UC       | ResNet50+ViT-B/16 | 36.67  | 32.63 | 33.44 |
|        UO         | ResNet50+ViT-B/16 | 37.65  | 33.61 | 34.28 |
|        UV         | ResNet50+ViT-B/16 | 29.23  | 33.95 | 33.29 |

You can download the pretrained models from this [link](https://postechackr-my.sharepoint.com/:u:/g/personal/sosfd_postech_ac_kr/ESOQa6xOkgNJpIuP_QSlth0BcdTtCYrIy0tAqdIsf422rg?e=C80ooS).

The downloaded files should be placed as follows:
```
LAIN
├── hicodet
│   └── hico_20160224_det
└── checkpoints 
    ├── pretrained_clip
    └── pretrained_detr
```
## Training & Evaluation
We provide four training and evaluation scripts for different zero-shot HOI detection settings (*i.e.*, RF-UC, NF-UC, UO and UV), all located in the ```scripts``` folder.

```shell
#training
bash scripts/training/[NF-UC,RF-UC,UO,UV].sh

#evaluation
bash scripts/eval/[NF-UC,RF-UC,UO,UV].sh
```

### Integrated adapters and SceneGate V5

The AutoDL scripts discover both pretrained files recursively under
`/root/Lain/checkpoints` and expect an already extracted dataset at
`/root/Lain/hico_20160224_det`. They only validate these assets; they never
download or extract them. Override the locations with `LAIN_PRETRAINED_ROOT`
or `HICO_IMAGE_ROOT` when needed.

Normal UO training writes checkpoints and WandB files below
`/root/autodl-tmp/Lain/UO_scene_text_obj_run`. To keep training after the SSH
client or local computer is closed, start the detached tmux launcher:

```shell
cd /root/Lain/LAIN-main
bash scripts/training/UO_background.sh
```

The launcher prints the exact checkpoint directory, timestamped log path,
tmux attach command, and `tail -f` command.

`scripts/training/UO.sh` enables the LAIN visual instance adapter, text semantic
adapter, object-conditioned adapter, and SceneGate V5 together. HICO candidates
follow the original LAIN rule: each detected human is paired with every other
detection, including another person.

All reported AP values use the model scores directly. There is no class-specific
suppression and no unseen-class score boost. With SceneGate diagnostics enabled,
the following files are written under `--output-dir`:

```text
scene_gate_diagnostics.json
scene_gate_diagnostics.csv
```

They contain Gate ON/OFF full, unseen, and seen mAP plus `raw`, `u`, `u_rel`, and
residual-ratio statistics. The pair/zero/shuffle/random ablation is available via:

```shell
python scripts/diagnostics/relative_gate_v5_diag.py \
  --checkpoint /path/to/checkpoint.pt \
  --output checkpoints/UO_scene_text_obj/scene_gate_ablation.json
```

Gate-only quick training from a complete checkpoint is available via:

```shell
bash scripts/diagnostics/relative_gate_v5_quick.sh /path/to/checkpoint.pt
```

## Acknowledgement
Our implementation is built upon [ADA-CM](https://github.com/ltttpku/ADA-CM) and [UPT](https://github.com/fredzzhang/upt).
We are grateful to the authors for their excellent work and for making their code publicly available.
