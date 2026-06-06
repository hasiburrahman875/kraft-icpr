# Install

Use the existing conda environment if available:

```bash
conda activate yolov12_botsort
```

Or install the Python dependencies into a CUDA-enabled PyTorch environment:

```bash
pip install -r requirements.txt
```

The reproduction package includes the required detector outputs, checkpoints, ReID weights, optional packaged MUAV checksum track files, and dataset folders used by the provided configs.

Run:

```bash
cd /cluster/pixstor/madrias-lab/Hasibur/AT/repro_outputs/kraft-uavswarm-repro
PYTHON=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python ./run.sh all
```

Dataset-specific commands are also available:

```bash
PYTHON=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python ./run.sh uavswarm
PYTHON=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python ./run.sh w2c
PYTHON=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python ./run.sh muav
```

For MUAV, `./run.sh muav` regenerates tracks from packaged detections before evaluation. The faster evaluator-only checksum is:

```bash
PYTHON=/home/mrpk9/.conda/envs/yolov12_botsort/bin/python ./run.sh muav-eval
```

## Support

If reproduction fails, please report the command, dataset target, log files under `outputs/<target>/logs/`, and any generated `results/<target>/evaluation/summary.tsv`. During anonymous review, use the official review discussion channel. For public code or asset issues, open a GitHub issue in the repository. After de-anonymization, contact the corresponding author listed in the paper.
