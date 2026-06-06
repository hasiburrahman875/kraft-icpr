# KRAfT ICPR Reproducibility Package

Official reproducibility package for the ICPR 2026 accepted paper:

```text
KRAfT: Kalman Residual Diffusion with Formation Awareness for UAV Swarm Tracking
Md. Hasibur Rahman and Sanjay Madria
Department of Computer Science
Missouri University of Science and Technology
Rolla, MO 65401, USA
```

This repository contains the code and configuration files for reproducing the KRAfT tracking runs on:

- UAVSwarm
- UAVSwarm-W2C
- MUAV

Large files are not stored in GitHub. Dataset files, detections, checkpoints, and optional checksum tracks are distributed separately through Zenodo.

## Quick Start

```bash
git clone https://github.com/hasiburrahman875/kraft-icpr.git
cd kraft-icpr

# Download kraft-uavswarm-assets.tar.gz from Zenodo into the repository root.
sha256sum -c kraft-uavswarm-assets.sha256
tar -xzf kraft-uavswarm-assets.tar.gz

conda activate yolov12_botsort
PYTHON=$(which python) ./run.sh all
```

The final summaries will be written to:

```text
results/uavswarm/evaluation/summary.tsv
results/uavswarm-w2c/evaluation/summary.tsv
results/muav/evaluation/summary.tsv
```

## Reproduction Steps

### 1. Clone

```bash
git clone https://github.com/hasiburrahman875/kraft-icpr.git
cd kraft-icpr
```

### 2. Download Assets

Download the asset archive from the public Zenodo record:

```text
kraft-uavswarm-assets.tar.gz
```

Zenodo DOI:

```text
https://doi.org/10.5281/zenodo.20566871
```

The archive contains:

```text
dataset/
detections/
checkpoints/
tracks/
```

Place `kraft-uavswarm-assets.tar.gz` in the repository root, next to `run.sh`.

### 3. Verify and Extract

Verify the archive:

```bash
sha256sum -c kraft-uavswarm-assets.sha256
```

Successful verification output:

```text
kraft-uavswarm-assets.tar.gz: OK
```

Extract:

```bash
tar -xzf kraft-uavswarm-assets.tar.gz
```

After extraction, confirm the required asset folders exist:

```bash
ls dataset detections checkpoints tracks
```

### 4. Install

Use a CUDA-enabled Python environment. The original experiments used:

```bash
conda activate yolov12_botsort
```

For a new environment, install the Python requirements:

```bash
pip install -r requirements.txt
```

The tested cluster Python path was:

```text
/home/mrpk9/.conda/envs/yolov12_botsort/bin/python
```

For non-cluster systems, set `PYTHON` to the Python executable from the active environment.

### 5. Run Reproduction

Run all datasets:

```bash
PYTHON=$(which python) ./run.sh all
```

Run one dataset:

```bash
PYTHON=$(which python) ./run.sh uavswarm
PYTHON=$(which python) ./run.sh w2c
PYTHON=$(which python) ./run.sh muav
```

Run the optional MUAV evaluator-only checksum:

```bash
PYTHON=$(which python) ./run.sh muav-eval
```

`./run.sh muav` is the main MUAV reproduction command. It regenerates tracks from packaged detections and then evaluates them. `./run.sh muav-eval` only evaluates packaged MUAV track files and is intended as a quick evaluator check.

### 6. Read Results

The runner creates intermediate files under:

```text
outputs/uavswarm/
outputs/uavswarm-w2c/
outputs/muav/
outputs/muav-eval-tracks/
```

Read the final summaries here:

```text
results/uavswarm/evaluation/summary.tsv
results/uavswarm-w2c/evaluation/summary.tsv
results/muav/evaluation/summary.tsv
results/muav-eval-tracks/evaluation/summary.tsv
```

## Repository Layout

Code and configs in GitHub:

```text
run.sh
reproduce.py
requirements.txt
config_uavswarm.yaml
config_w2c.yaml
config_muav.yaml
config_muav_eval_tracks.yaml
kraft/
```

Assets from Zenodo after extraction:

```text
dataset/
detections/
checkpoints/
tracks/
```

## Checkpoint Paths

The YAML configs use explicit checkpoint paths.

| Dataset/fold | KRAfT/DiffMOT checkpoint | ReID checkpoint |
|---|---|---|
| UAVSwarm | `checkpoints/uavswarm/kraft_uavswarm.pt` | `checkpoints/uavswarm/reid_uavswarm.pth` |
| UAVSwarm-W2C fold1 | `checkpoints/uavswarm-w2c/fold1/kraft_uavswarm_w2c_fold1.pt` | `checkpoints/uavswarm-w2c/fold1/reid_uavswarm_w2c_fold1.pth` |
| UAVSwarm-W2C fold2 | `checkpoints/uavswarm-w2c/fold2/kraft_uavswarm_w2c_fold2.pt` | `checkpoints/uavswarm-w2c/fold2/reid_uavswarm_w2c_fold2.pth` |
| UAVSwarm-W2C fold3 | `checkpoints/uavswarm-w2c/fold3/kraft_uavswarm_w2c_fold3.pt` | `checkpoints/uavswarm-w2c/fold3/reid_uavswarm_w2c_fold3.pth` |
| MUAV fold1 | `checkpoints/muav/fold1/kraft_muav_fold1.pt` | `checkpoints/muav/fold1/reid_muav_fold1.pth` |
| MUAV fold2 | `checkpoints/muav/fold2/kraft_muav_fold2.pt` | `checkpoints/muav/fold2/reid_muav_fold2.pth` |
| MUAV fold3 | `checkpoints/muav/fold3/kraft_muav_fold3.pt` | `checkpoints/muav/fold3/reid_muav_fold3.pth` |

Some checkpoint files are byte-identical named copies of the same source weights. They are kept fold-specific so each config is self-contained.

## Notes

UAVSwarm settings are specified in `config_uavswarm.yaml`.

UAVSwarm-W2C uses one configuration per fold in `config_w2c.yaml`.

MUAV uses one configuration per fold in `config_muav.yaml`. The default MUAV config sets `w_assoc_emb: 0.0`, so ReID embeddings are not used in the main MUAV tracking run, although fold-wise ReID files are included for completeness.

For W2C, `fold1/Swarm-77` declares `seqLength=812` but contains 811 JPG images in the packaged dataset. The runner uses all available images and caps MOT GT/detection files to the valid sequence length before TrackEval.

For MUAV, the packaged fold folders do not include `seqinfo.ini`. The runner generates minimal MOT `seqinfo.ini` files from the packaged images and caps GT rows to the available image count before TrackEval.

## Troubleshooting

If a run fails, first verify the asset folders and checksum:

```bash
ls dataset detections checkpoints tracks
sha256sum -c kraft-uavswarm-assets.sha256
```

Then inspect the relevant log directory:

```text
outputs/<target>/logs/
```

Issue reports should include:

- the executed command
- the target dataset: `uavswarm`, `w2c`, `muav`, or `muav-eval`
- the relevant log file from `outputs/<target>/logs/`
- the generated `results/<target>/evaluation/summary.tsv`, if it exists
- Python, CUDA, PyTorch, and GPU details

During anonymous review, use the official review discussion channel. For public code or asset issues, open a GitHub issue. After de-anonymization, contact the corresponding author listed in the paper.
