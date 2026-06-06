# KRAfT ICPR 2026 Reproducibility Package

This repository is the official reproducibility package for the ICPR 2026 accepted paper:

```text
KRAfT: Kalman Residual Diffusion with Formation Awareness for UAV Swarm Tracking
Md. Hasibur Rahman and Sanjay Madria
Department of Computer Science
Missouri University of Science and Technology
Rolla, MO 65401, USA
```

The repository contains the source code, configuration files, and evaluation runner. The large assets required for reproduction are hosted on Zenodo.

## Artifact Structure

The GitHub repository contains:

```text
run.sh
reproduce.py
requirements.txt
download_assets.sh
config_uavswarm.yaml
config_w2c.yaml
config_muav.yaml
config_muav_eval_tracks.yaml
kraft/
```

The Zenodo archive provides:

```text
dataset/
detections/
checkpoints/
tracks/
```

After setup, the repository root should contain both groups of files.

## Requirements

- Linux environment
- CUDA-capable GPU
- Conda or another Python environment manager
- Python 3.10 recommended
- At least 50 GB of free disk space for the compressed archive and extracted assets

The reproduction code expects precomputed detections and checkpoints from the Zenodo archive. Detector training and detector inference are not part of this artifact.

## Step 1: Clone the Repository

```bash
git clone https://github.com/hasiburrahman875/kraft-icpr.git
cd kraft-icpr
```

All commands below assume the current working directory is the repository root.

## Step 2: Create the Environment

```bash
conda create -n kraft-icpr python=3.10 -y
conda activate kraft-icpr
pip install -r requirements.txt
```

If a compatible CUDA-enabled PyTorch environment already exists, it can be used instead. In that case, activate the environment and install the missing packages from `requirements.txt`.

## Step 3: Download and Extract Assets

Assets are available from Zenodo:

```text
https://doi.org/10.5281/zenodo.20566871
```

### Recommended Download

Run the provided helper script from the repository root:

```bash
bash download_assets.sh
```

The script downloads:

```text
kraft-uavswarm-assets.tar.gz
kraft-uavswarm-assets.sha256
```

It then verifies the SHA256 checksum and extracts the archive into the repository root.

### Manual Download

The files can also be downloaded manually:

```bash
curl -L -o kraft-uavswarm-assets.sha256 \
  https://zenodo.org/api/records/20566871/files/kraft-uavswarm-assets.sha256/content

curl -L -o kraft-uavswarm-assets.tar.gz \
  https://zenodo.org/api/records/20566871/files/kraft-uavswarm-assets.tar.gz/content

sha256sum -c kraft-uavswarm-assets.sha256
tar -xzf kraft-uavswarm-assets.tar.gz
```

The checksum verification should report:

```text
kraft-uavswarm-assets.tar.gz: OK
```

After extraction, verify the expected asset folders:

```bash
ls dataset detections checkpoints tracks
```

## Step 4: Run the Reproduction

Run all datasets:

```bash
PYTHON=$(which python) ./run.sh all
```

Run individual datasets:

```bash
PYTHON=$(which python) ./run.sh uavswarm
PYTHON=$(which python) ./run.sh w2c
PYTHON=$(which python) ./run.sh muav
```

Optional evaluator-only MUAV checksum:

```bash
PYTHON=$(which python) ./run.sh muav-eval
```

The `muav` target regenerates tracks from packaged detections before evaluation. The `muav-eval` target evaluates packaged MUAV track files and is provided only as a faster evaluator check.

## Step 5: Locate Outputs

Intermediate tracking and evaluation files are written under:

```text
outputs/uavswarm/
outputs/uavswarm-w2c/
outputs/muav/
outputs/muav-eval-tracks/
```

Final summary files are written under:

```text
results/uavswarm/evaluation/summary.tsv
results/uavswarm-w2c/evaluation/summary.tsv
results/muav/evaluation/summary.tsv
results/muav-eval-tracks/evaluation/summary.tsv
```

## Configuration Files

The reproduction runner uses one YAML configuration per benchmark group:

```text
config_uavswarm.yaml
config_w2c.yaml
config_muav.yaml
config_muav_eval_tracks.yaml
```

UAVSwarm-W2C and MUAV use fold-specific checkpoint paths. The main checkpoint layout is:

| Dataset/fold | KRAfT/DiffMOT checkpoint | ReID checkpoint |
|---|---|---|
| UAVSwarm | `checkpoints/uavswarm/kraft_uavswarm.pt` | `checkpoints/uavswarm/reid_uavswarm.pth` |
| UAVSwarm-W2C fold1 | `checkpoints/uavswarm-w2c/fold1/kraft_uavswarm_w2c_fold1.pt` | `checkpoints/uavswarm-w2c/fold1/reid_uavswarm_w2c_fold1.pth` |
| UAVSwarm-W2C fold2 | `checkpoints/uavswarm-w2c/fold2/kraft_uavswarm_w2c_fold2.pt` | `checkpoints/uavswarm-w2c/fold2/reid_uavswarm_w2c_fold2.pth` |
| UAVSwarm-W2C fold3 | `checkpoints/uavswarm-w2c/fold3/kraft_uavswarm_w2c_fold3.pt` | `checkpoints/uavswarm-w2c/fold3/reid_uavswarm_w2c_fold3.pth` |
| MUAV fold1 | `checkpoints/muav/fold1/kraft_muav_fold1.pt` | `checkpoints/muav/fold1/reid_muav_fold1.pth` |
| MUAV fold2 | `checkpoints/muav/fold2/kraft_muav_fold2.pt` | `checkpoints/muav/fold2/reid_muav_fold2.pth` |
| MUAV fold3 | `checkpoints/muav/fold3/kraft_muav_fold3.pt` | `checkpoints/muav/fold3/reid_muav_fold3.pth` |

Some checkpoint files are byte-identical named copies of the same source weights. They are kept fold-specific so each configuration is self-contained.

## Implementation Notes

UAVSwarm-W2C uses one configuration per fold in `config_w2c.yaml`.

MUAV uses one configuration per fold in `config_muav.yaml`. The default MUAV config sets `w_assoc_emb: 0.0`, so ReID embeddings are not used in the main MUAV tracking run, although fold-wise ReID files are included for completeness.

For MUAV, the packaged fold folders do not include `seqinfo.ini`. Minimal MOT `seqinfo.ini` files from the packaged images and caps GT rows to the available image count before TrackEval.

## Troubleshooting

First, verify that the assets were extracted correctly:

```bash
ls dataset detections checkpoints tracks
sha256sum -c kraft-uavswarm-assets.sha256
```

Then inspect the relevant log directory:

```text
outputs/<target>/logs/
```

Issue reports should include:

- executed command
- target dataset: `uavswarm`, `w2c`, `muav`, or `muav-eval`
- relevant log file from `outputs/<target>/logs/`
- generated `results/<target>/evaluation/summary.tsv`, if available
- Python, CUDA, PyTorch, and GPU details

During anonymous review, use the official review discussion channel. For public code or asset issues, open a GitHub issue. After de-anonymization, contact the corresponding author listed in the paper.
