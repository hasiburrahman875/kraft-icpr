# Installation

This repository is the reproducibility package for the ICPR 2026 accepted paper:

```text
KRAfT: Kalman Residual Diffusion with Formation Awareness for UAV Swarm Tracking
Md. Hasibur Rahman and Sanjay Madria
Department of Computer Science
Missouri University of Science and Technology
Rolla, MO 65401, USA
```

## Environment

Create and activate a CUDA-enabled Python environment:

```bash
conda create -n kraft-icpr python=3.10 -y
conda activate kraft-icpr
pip install -r requirements.txt
```

## Assets

The required datasets, detections, checkpoints, and optional checksum tracks are hosted on Zenodo:

```text
https://doi.org/10.5281/zenodo.20566871
```

From the repository root, download and extract the assets with:

```bash
bash download_assets.sh
```

Manual download commands are provided in `README.md`.

## Execution

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

Run the optional MUAV evaluator-only checksum:

```bash
PYTHON=$(which python) ./run.sh muav-eval
```

Final summaries are written under:

```text
results/<target>/evaluation/summary.tsv
```

## Support

Issue reports should include the executed command, dataset target, log files under `outputs/<target>/logs/`, and any generated `results/<target>/evaluation/summary.tsv`. During anonymous review, use the official review discussion channel. For public code or asset issues, open a GitHub issue in the repository. After de-anonymization, contact the corresponding author listed in the paper.
