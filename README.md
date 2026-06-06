# KRAfT Reproduction for UAVSwarm, UAVSwarm-W2C, and MUAV

This repository provides one reproducible runner for the KRAfT tracking results on three datasets:

- UAVSwarm
- UAVSwarm-W2C
- MUAV

The same entry point, `run.sh`, calls `reproduce.py` for all datasets. Dataset-specific behavior is controlled only by YAML configs.

## Repository Contents

The code repository should contain:

```text
README.md
INSTALL.md
requirements.txt
run.sh
reproduce.py
config_uavswarm.yaml
config_w2c.yaml
config_muav.yaml
config_muav_eval_tracks.yaml
kraft/
```

The large assets are distributed separately through Zenodo:

```text
dataset/
detections/
checkpoints/
tracks/
```

The Zenodo draft created for these assets is:

```text
10.5281/zenodo.20566871
```

If the record is still unpublished, use the Zenodo private preview/download link provided by the authors. After publication, use the public Zenodo record download link.

## Assets

The asset archive is:

```text
kraft-uavswarm-assets.tar.gz
```

Expected SHA256:

```text
10a136c11afab504e86df6149aa5ad8b47df7eeb67405d70751c676c4426b94e  kraft-uavswarm-assets.tar.gz
```

After downloading the archive into the repository root, verify and extract it:

```bash
sha256sum -c kraft-uavswarm-assets.sha256
tar -xzf kraft-uavswarm-assets.tar.gz
```

After extraction, the repository root should contain:

```text
dataset/
detections/
checkpoints/
tracks/
```

## Checkpoints

The configs use explicit dataset and fold-specific checkpoint paths.

| Dataset/fold | KRAfT/DiffMOT checkpoint | ReID checkpoint |
|---|---|---|
| UAVSwarm | `checkpoints/uavswarm/kraft_uavswarm.pt` | `checkpoints/uavswarm/reid_uavswarm.pth` |
| UAVSwarm-W2C fold1 | `checkpoints/uavswarm-w2c/fold1/kraft_uavswarm_w2c_fold1.pt` | `checkpoints/uavswarm-w2c/fold1/reid_uavswarm_w2c_fold1.pth` |
| UAVSwarm-W2C fold2 | `checkpoints/uavswarm-w2c/fold2/kraft_uavswarm_w2c_fold2.pt` | `checkpoints/uavswarm-w2c/fold2/reid_uavswarm_w2c_fold2.pth` |
| UAVSwarm-W2C fold3 | `checkpoints/uavswarm-w2c/fold3/kraft_uavswarm_w2c_fold3.pt` | `checkpoints/uavswarm-w2c/fold3/reid_uavswarm_w2c_fold3.pth` |
| MUAV fold1 | `checkpoints/muav/fold1/kraft_muav_fold1.pt` | `checkpoints/muav/fold1/reid_muav_fold1.pth` |
| MUAV fold2 | `checkpoints/muav/fold2/kraft_muav_fold2.pt` | `checkpoints/muav/fold2/reid_muav_fold2.pth` |
| MUAV fold3 | `checkpoints/muav/fold3/kraft_muav_fold3.pt` | `checkpoints/muav/fold3/reid_muav_fold3.pth` |

Some checkpoint files are byte-identical named copies of the same source weights. They are intentionally stored with dataset/fold-specific names so that each config is self-contained.

## Installation

Use the existing conda environment if available:

```bash
conda activate yolov12_botsort
```

Or install dependencies in a CUDA-enabled PyTorch environment:

```bash
pip install -r requirements.txt
```

On the original cluster, the tested Python executable was:

```bash
/home/mrpk9/.conda/envs/yolov12_botsort/bin/python
```

For another machine, replace `PYTHON=...` with the path to your environment's Python.

## Run

From the repository root:

```bash
PYTHON=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python ./run.sh uavswarm
PYTHON=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python ./run.sh w2c
PYTHON=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python ./run.sh muav
```

To run all three:

```bash
PYTHON=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python ./run.sh all
```

The MUAV command regenerates tracks from packaged detections before evaluation. It does not evaluate only saved final summaries.

An optional fast MUAV evaluator checksum is available:

```bash
PYTHON=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python ./run.sh muav-eval
```

This checksum evaluates the packaged MUAV track files in `tracks/muav/fold*`. It is useful for verifying the evaluator quickly, but the main reviewer reproduction path is `./run.sh muav`.

## Outputs

Fresh runs write intermediate files under:

```text
outputs/uavswarm/
outputs/uavswarm-w2c/
outputs/muav/
outputs/muav-eval-tracks/
```

Final summaries are copied to:

```text
results/uavswarm/evaluation/summary.tsv
results/uavswarm-w2c/evaluation/summary.tsv
results/muav/evaluation/summary.tsv
results/muav-eval-tracks/evaluation/summary.tsv
```

## Config Notes

UAVSwarm uses the two documented non-oracle settings in `config_uavswarm.yaml`.

UAVSwarm-W2C uses one configuration per fold. The fold definitions and fold-wise checkpoints are in `config_w2c.yaml`.

MUAV uses one configuration per fold. The default MUAV command uses `run_mode: track` with `tracker_impl: diffmot`, fold-wise checkpoints under `checkpoints/muav/fold*/`, and detector files under `detections/muav-diffmot/fold*`. MUAV ReID files are packaged for completeness, but the default MUAV config sets `w_assoc_emb: 0.0`, so ReID embeddings are not used in the reported MUAV tracking run.

For W2C, `fold1/Swarm-77` declares `seqLength=812` but contains 811 JPG images in the bundled dataset. The runner uses all available images and caps MOT GT/det files to the declared `seqLength` before TrackEval.

For MUAV, the packaged fold folders do not include `seqinfo.ini`; the runner generates minimal MOT `seqinfo.ini` files from the bundled images and caps GT rows to the available image count before TrackEval.

## Support

If you encounter an issue while reproducing the results, please include the following information in your message:

- the command you ran
- the dataset target, for example `uavswarm`, `w2c`, `muav`, or `muav-eval`
- the relevant log files from `outputs/<target>/logs/`
- the generated summary file from `results/<target>/evaluation/summary.tsv`, if it exists
- your Python, CUDA, PyTorch, and GPU details

During anonymous review, please use the official review discussion channel. For public code or asset issues, open a GitHub issue in the repository. After de-anonymization, contact the corresponding author listed in the paper.
