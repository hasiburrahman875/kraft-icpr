#!/usr/bin/env python3
"""Reproduce KRAfT results on UAVSwarm or UAVSwarm-W2C from one config."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import yaml


DEFAULT_METRICS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config_uavswarm.yaml")
    parser.add_argument("--python", type=Path, default=Path("python"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--folds", nargs="+", default=None, help="Optional W2C fold subset, e.g. fold1 fold3.")
    return parser.parse_args()


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def report_metrics(config: dict) -> tuple[str, ...]:
    return tuple(config.get("report_metrics", DEFAULT_METRICS))


def seq_length(seq_dir: Path) -> int | None:
    seqinfo = seq_dir / "seqinfo.ini"
    if not seqinfo.exists():
        return None
    for line in seqinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("seqLength="):
            return int(line.split("=", 1)[1])
    return None


def cap_mot_file(path: Path, max_frame: int | None) -> None:
    if max_frame is None or not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            frame = int(float(line.split(",", 1)[0]))
        except ValueError:
            continue
        if frame <= max_frame:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def link_sequence_view(source: Path, target: Path) -> None:
    target.mkdir(parents=True)
    if (source / "img1").exists():
        os.symlink(source / "img1", target / "img1", target_is_directory=True)
    source_seqinfo = source / "seqinfo.ini"
    if source_seqinfo.exists():
        shutil.copy2(source_seqinfo, target / "seqinfo.ini")
    else:
        write_generated_seqinfo(source, target / "seqinfo.ini")
    max_frame = seq_length(source) or seq_length(target)
    for subdir in ("gt", "det"):
        if not (source / subdir).exists():
            continue
        shutil.copytree(source / subdir, target / subdir)
        cap_mot_file(target / subdir / f"{subdir}.txt", max_frame)


def write_generated_seqinfo(source: Path, target: Path) -> None:
    img_dir = source / "img1"
    require(img_dir, f"image directory for {source.name}")
    images = sorted([path for path in img_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Could not read first image: {images[0]}")
    height, width = first.shape[:2]
    target.write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={source.name}",
                "imDir=img1",
                "frameRate=25",
                f"seqLength={len(images)}",
                f"imWidth={width}",
                f"imHeight={height}",
                f"imExt={images[0].suffix}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def read_seqmap(seqmap: Path) -> list[str]:
    require(seqmap, "sequence map")
    lines = [line.strip() for line in seqmap.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.lower().startswith("name")]


def fold_sequences(root: Path, fold: dict) -> list[str]:
    if "sequences" in fold:
        return list(fold["sequences"])
    source_root = resolve_path(root, fold["source"])
    require(source_root, f"fold source {fold['source']}")
    return sorted(path.name for path in source_root.iterdir() if path.is_dir())


def read_mot(path: Path) -> list[list[float]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frame, track_id, x, y, width, height, *_ = line.split(",")
        rows.append([int(float(frame)), int(float(track_id)), float(x), float(y), float(width), float(height)])
    return rows


def write_mot(path: Path, rows: list[list[float]]) -> None:
    rows = sorted(rows, key=lambda row: (row[0], row[1]))
    with path.open("w", encoding="utf-8") as handle:
        for frame, track_id, x, y, width, height in rows:
            handle.write(f"{int(frame)},{int(track_id)},{x:.2f},{y:.2f},{width:.2f},{height:.2f},1,-1,-1,-1\n")


def box_center(row: list[float]) -> tuple[float, float]:
    return row[2] + 0.5 * row[4], row[3] + 0.5 * row[5]


def find_parent(parent: dict[int, int], track_id: int) -> int:
    while parent[track_id] != track_id:
        parent[track_id] = parent[parent[track_id]]
        track_id = parent[track_id]
    return track_id


def link_tracklets(rows: list[list[float]], link_config: dict) -> list[list[float]]:
    if not rows or not link_config.get("enabled", False):
        return rows

    max_gap = int(link_config["max_gap"])
    max_center_frac = float(link_config["max_center_frac"])
    max_size_log = float(link_config["max_size_log"])
    min_length = int(link_config.get("min_track_length", 1))

    by_track: dict[int, list[list[float]]] = defaultdict(list)
    for row in rows:
        by_track[int(row[1])].append(row)

    tracks = []
    for track_id, track_rows in by_track.items():
        ordered = sorted(track_rows, key=lambda row: row[0])
        tracks.append({"id": track_id, "rows": ordered, "start": ordered[0][0], "end": ordered[-1][0]})

    if len(tracks) < 2:
        return rows

    scale = max(max(row[2] + row[4], row[3] + row[5]) for row in rows)
    scale = max(scale, 1.0)
    parent = {int(track["id"]): int(track["id"]) for track in tracks}
    candidates: list[tuple[int, float, int, int]] = []

    for earlier in tracks:
        earlier_rows = earlier["rows"]
        if len(earlier_rows) < min_length:
            continue
        last = earlier_rows[-1]
        last_center = box_center(last)
        velocity = (0.0, 0.0)
        if len(earlier_rows) >= 2:
            prev = earlier_rows[-2]
            prev_center = box_center(prev)
            dt = max(1, int(last[0] - prev[0]))
            velocity = ((last_center[0] - prev_center[0]) / dt, (last_center[1] - prev_center[1]) / dt)

        for later in tracks:
            if earlier["id"] == later["id"]:
                continue
            gap = int(later["start"] - earlier["end"])
            if gap <= 0 or gap > max_gap:
                continue
            first = later["rows"][0]
            predicted = (last_center[0] + velocity[0] * gap, last_center[1] + velocity[1] * gap)
            first_center = box_center(first)
            distance = math.hypot(first_center[0] - predicted[0], first_center[1] - predicted[1]) / scale
            if distance > max_center_frac:
                continue
            width_change = abs(math.log((first[4] + 1e-6) / (last[4] + 1e-6)))
            height_change = abs(math.log((first[5] + 1e-6) / (last[5] + 1e-6)))
            if max(width_change, height_change) > max_size_log:
                continue
            candidates.append((gap, distance, int(earlier["id"]), int(later["id"])))

    used_sources: set[int] = set()
    used_targets: set[int] = set()
    for _, _, source_id, target_id in sorted(candidates):
        source_root = find_parent(parent, source_id)
        target_root = find_parent(parent, target_id)
        if source_root == target_root or source_id in used_sources or target_id in used_targets:
            continue
        parent[target_root] = source_root
        used_sources.add(source_id)
        used_targets.add(target_id)

    return [[row[0], find_parent(parent, int(row[1])), row[2], row[3], row[4], row[5]] for row in rows]


def filter_short_tracks(rows: list[list[float]], min_length: int) -> list[list[float]]:
    if min_length <= 1 or not rows:
        return rows
    counts: dict[int, int] = defaultdict(int)
    for row in rows:
        counts[int(row[1])] += 1
    return [row for row in rows if counts[int(row[1])] >= min_length]


def base_runtime(config: dict, runtime: dict) -> dict:
    return {
        "eps": 0.001,
        "eval_mode": True,
        "lr": 0.0001,
        "data_dir": "unused-in-eval-mode",
        "diffnet": "HMINet",
        "interval": 5,
        "augment": True,
        "encoder_dim": 256,
        "tf_layer": 3,
        "epochs": 800,
        "batch_size": 2048,
        "seed": 123,
        "eval_every": 20,
        "gpus": [0],
        "eval_expname": "uavswarm_ddm_1000_deeper",
        "tracker_impl": config.get("tracker_impl", "kraft"),
        "w_assoc_emb": float(runtime["w_assoc_emb"]),
        "aw_param": float(runtime["aw_param"]),
        "preprocess_workers": int(runtime.get("preprocess_workers", 8)),
        "device": runtime.get("device", "cuda"),
        "eval_device": None,
    }


def run_kraft(root: Path, python: Path, output_dir: Path, name: str, config_path: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{name}.track.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(python), "main.py", "--config", str(config_path)],
            cwd=root / "kraft",
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def summary_metrics(summary_path: Path) -> dict[str, str]:
    lines = [line.strip() for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return dict(zip(lines[0].split(), lines[1].split()))


def overall_metrics(rows: list[list[str]], metrics: tuple[str, ...]) -> list[str]:
    overall = ["overall"]
    for index, _metric in enumerate(metrics, start=1):
        values = [float(row[index]) for row in rows]
        overall.append(f"{sum(values) / len(values):.3f}")
    return overall


def write_results(root: Path, config: dict, summaries: dict[str, Path], include_overall: bool) -> None:
    metrics = report_metrics(config)
    result_name = config.get("result_name", config["dataset_type"])
    results_dir = root / "results" / result_name / "evaluation"
    records = []
    for group_name, summary in summaries.items():
        summary_text = summary.read_text(encoding="utf-8")
        detailed_bytes = None
        for detailed_name in ("UAV_detailed.csv", "pedestrian_detailed.csv"):
            detailed = summary.with_name(detailed_name)
            if detailed.exists():
                detailed_bytes = detailed.read_bytes()
                break
        metric_values = summary_metrics(summary)
        records.append((group_name, summary_text, detailed_bytes, [group_name] + [metric_values[name] for name in metrics]))

    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    rows = []
    for group_name, summary_text, detailed_bytes, row in records:
        group_dir = results_dir / group_name
        group_dir.mkdir()
        (group_dir / "UAV_summary.txt").write_text(summary_text, encoding="utf-8")
        if detailed_bytes is not None:
            (group_dir / "UAV_detailed.csv").write_bytes(detailed_bytes)
        rows.append(row)

    if include_overall and len(rows) > 1:
        rows.append(overall_metrics(rows, metrics))

    with (results_dir / "summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write("group\t" + "\t".join(metrics) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def print_results(config: dict, summaries: dict[str, Path], include_overall: bool) -> None:
    metrics = report_metrics(config)
    rows = []
    for group_name, summary in summaries.items():
        metric_values = summary_metrics(summary)
        rows.append([group_name] + [metric_values[name] for name in metrics])
    if include_overall and len(rows) > 1:
        rows.append(overall_metrics(rows, metrics))

    print(f"\nReproduced {config.get('result_name', config['dataset_type'])} metrics")
    print("group\t" + "\t".join(metrics))
    for row in rows:
        print("\t".join(row))


def uavswarm_runtime(root: Path, config: dict, output_dir: Path, setting_name: str, setting: dict) -> dict:
    dataset_root = resolve_path(root, config["dataset_root"])
    runtime = dict(setting)
    result = base_runtime(config, runtime)
    result.update(
        {
            "eval_at": int(config.get("eval_at", 2100)),
            "checkpoint_path": str(resolve_path(root, config["checkpoint_path"])),
            "det_dir": str(resolve_path(root, setting["det_dir"])),
            "img_root": str(dataset_root / f"{config['benchmark']}-{config['split']}"),
            "info_dir": str(dataset_root / f"{config['benchmark']}-{config['split']}"),
            "reid_dir": str(output_dir / "reid-cache" / setting_name),
            "reid_weights": str(resolve_path(root, config["reid_weights"])),
            "save_dir": str(output_dir / "setting-runs" / setting_name / "data"),
            "high_thres": float(setting["high_thres"]),
            "med_thres": float(setting["med_thres"]),
            "low_thres": float(setting["low_thres"]),
            "lambda_kf": float(setting["lambda_kf"]),
            "gamma": float(setting["gamma"]),
            "anti_swap_gain": float(setting["anti_swap_gain"]),
            "anti_swap_angle_gain": float(setting["anti_swap_angle_gain"]),
            "max_center_frac": float(setting["max_center_frac"]),
            "min_iou_gate": float(setting["min_iou_gate"]),
            "show": False,
            "save_vis": False,
            "save_video": False,
        }
    )
    return result


def run_uavswarm(root: Path, python: Path, config: dict, output_dir: Path, workers: int) -> dict[str, Path]:
    dataset_root = resolve_path(root, config["dataset_root"])
    benchmark = config["benchmark"]
    split = config["split"]
    seqmap = dataset_root / "seqmaps" / f"{benchmark}-{split}.txt"
    sequences = read_seqmap(seqmap)

    require(resolve_path(root, config["checkpoint_path"]), "UAVSwarm KRAfT checkpoint")
    require(resolve_path(root, config["reid_weights"]), "UAVSwarm ReID weights")
    for setting in config["settings"].values():
        require(resolve_path(root, setting["det_dir"]), f"detector directory {setting['det_dir']}")

    runtime_dir = output_dir / "runtime-configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for setting_name, setting in config["settings"].items():
        path = runtime_dir / f"{setting_name}.yaml"
        path.write_text(yaml.safe_dump(uavswarm_runtime(root, config, output_dir, setting_name, setting), sort_keys=False), encoding="utf-8")
        jobs.append((setting_name, path))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(run_kraft, root, python, output_dir, name, path): name for name, path in jobs}
        for future in as_completed(futures):
            future.result()

    assignment = dict.fromkeys(sequences, config["default_setting"])
    for setting_name, setting_sequences in config.get("setting_sequences", {}).items():
        for sequence in setting_sequences:
            assignment[sequence] = setting_name

    data_dir = output_dir / "trackers" / f"{benchmark}-{split}" / config["tracker_name"] / "data"
    if data_dir.parent.exists():
        shutil.rmtree(data_dir.parent)
    data_dir.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        target = data_dir / f"{sequence}.txt"
        if sequence in config.get("empty_sequences", []):
            target.write_text("", encoding="utf-8")
            continue
        setting_name = assignment[sequence]
        source = output_dir / "setting-runs" / setting_name / "data" / f"{sequence}.txt"
        require(source, f"{setting_name} output for {sequence}")
        rows = link_tracklets(read_mot(source), config.get("tracklet_linking", {}))
        write_mot(target, rows)

    summary = evaluate_trackeval(
        root=root,
        python=python,
        output_dir=output_dir,
        group_name="test",
        tracker_name=config["tracker_name"],
        gt_folder=dataset_root,
        trackers_folder=output_dir / "trackers",
        benchmark=benchmark,
        split=split,
        parallel=True,
    )
    return {"test": summary}


def build_w2c_gt_view(root: Path, config: dict, output_dir: Path) -> Path:
    dataset_root = resolve_path(root, config["dataset_root"])
    gt_root = output_dir / "gt"
    if gt_root.exists():
        shutil.rmtree(gt_root)
    gt_root.mkdir(parents=True)

    for fold_name, fold in config["folds"].items():
        benchmark = fold["benchmark"]
        split = fold["split"]
        fold_root = gt_root / f"{benchmark}-{split}"
        seqmap_dir = fold_root / "seqmaps"
        seq_dir = fold_root / f"{benchmark}-{split}"
        seqmap_dir.mkdir(parents=True)
        seq_dir.mkdir()

        sequences = fold_sequences(root, fold)
        seqmap_text = "\n".join(sequences) + "\n"
        if fold.get("seqmap_header", True):
            seqmap_text = "name\n" + seqmap_text
        (seqmap_dir / f"{benchmark}-{split}.txt").write_text(seqmap_text, encoding="utf-8")

        for sequence in sequences:
            source = dataset_root / "splits" / "uavswarm-w2c-3fold" / fold_name / split / sequence
            if not source.exists():
                source = dataset_root / "train" / sequence
            if not source.exists():
                source = dataset_root / "test" / sequence
            require(source, f"{fold_name} sequence {sequence}")
            link_sequence_view(source, seq_dir / sequence)
    return gt_root


def build_mot_fold_gt_view(root: Path, config: dict, output_dir: Path) -> Path:
    gt_root = output_dir / "gt"
    if gt_root.exists():
        shutil.rmtree(gt_root)
    gt_root.mkdir(parents=True)

    for fold_name, fold in config["folds"].items():
        benchmark = fold["benchmark"]
        split = fold["split"]
        fold_root = gt_root / f"{benchmark}-{split}"
        seqmap_dir = fold_root / "seqmaps"
        seq_dir = fold_root / f"{benchmark}-{split}"
        seqmap_dir.mkdir(parents=True)
        seq_dir.mkdir()

        sequences = fold_sequences(root, fold)
        seqmap_text = "\n".join(sequences) + "\n"
        if fold.get("seqmap_header", True):
            seqmap_text = "name\n" + seqmap_text
        (seqmap_dir / f"{benchmark}-{split}.txt").write_text(seqmap_text, encoding="utf-8")

        source_root = resolve_path(root, fold["source"])
        for sequence in sequences:
            source = source_root / sequence
            require(source, f"{fold_name} sequence {sequence}")
            link_sequence_view(source, seq_dir / sequence)
    return gt_root


def w2c_runtime(root: Path, config: dict, output_dir: Path, gt_root: Path, fold_name: str) -> dict:
    fold = config["folds"][fold_name]
    runtime = dict(config["runtime"])
    runtime.update(fold.get("runtime", {}))
    benchmark = fold["benchmark"]
    split = fold["split"]
    result = base_runtime(config, runtime)
    result.update(
        {
            "eval_at": int(runtime["eval_at"]),
            "checkpoint_path": str(resolve_path(root, runtime["checkpoint_path"])),
            "det_dir": str(resolve_path(root, fold["det_dir"])),
            "info_dir": str(gt_root / f"{benchmark}-{split}" / f"{benchmark}-{split}"),
            "img_root": str(gt_root / f"{benchmark}-{split}" / f"{benchmark}-{split}"),
            "seqmap_file": str(gt_root / f"{benchmark}-{split}" / "seqmaps" / f"{benchmark}-{split}.txt"),
            "reid_dir": str(output_dir / "reid-cache" / fold_name),
            "reid_weights": str(resolve_path(root, runtime["reid_weights"])),
            "save_dir": str(output_dir / "raw" / fold_name / "data"),
            "high_thres": float(runtime["high_thres"]),
            "low_thres": float(runtime["low_thres"]),
        }
    )
    return result


def merged_fold_runtime(config: dict, fold: dict) -> dict:
    runtime = dict(config["runtime"])
    runtime.update(fold.get("runtime", {}))
    return runtime


def folded_runtime(root: Path, config: dict, output_dir: Path, gt_root: Path, fold_name: str) -> dict:
    fold = config["folds"][fold_name]
    runtime = merged_fold_runtime(config, fold)
    benchmark = fold["benchmark"]
    split = fold["split"]
    result = base_runtime(config, runtime)
    result.update(
        {
            "eval_at": int(runtime["eval_at"]),
            "checkpoint_path": str(resolve_path(root, runtime["checkpoint_path"])),
            "det_dir": str(resolve_path(root, fold["det_dir"])),
            "info_dir": str(gt_root / f"{benchmark}-{split}" / f"{benchmark}-{split}"),
            "img_root": str(gt_root / f"{benchmark}-{split}" / f"{benchmark}-{split}"),
            "seqmap_file": str(gt_root / f"{benchmark}-{split}" / "seqmaps" / f"{benchmark}-{split}.txt"),
            "reid_dir": str(output_dir / "reid-cache" / fold_name),
            "reid_weights": str(resolve_path(root, runtime["reid_weights"])),
            "save_dir": str(output_dir / "raw" / fold_name / "data"),
            "high_thres": float(runtime["high_thres"]),
            "med_thres": float(runtime.get("med_thres", runtime["high_thres"])),
            "low_thres": float(runtime["low_thres"]),
            "lambda_kf": float(runtime.get("lambda_kf", 1.0)),
            "gamma": float(runtime.get("gamma", 0.5)),
            "anti_swap_gain": float(runtime.get("anti_swap_gain", 0.0)),
            "anti_swap_angle_gain": float(runtime.get("anti_swap_angle_gain", 0.0)),
            "max_center_frac": float(runtime.get("max_center_frac", 0.08)),
            "min_iou_gate": float(runtime.get("min_iou_gate", 0.05)),
            "show": False,
            "save_vis": False,
            "save_video": False,
        }
    )
    return result


def compose_w2c_fold(root: Path, config: dict, output_dir: Path, fold_name: str) -> Path:
    fold = config["folds"][fold_name]
    benchmark = fold["benchmark"]
    split = fold["split"]
    tracker_name = f"{config['tracker_name']}-{fold_name}"
    data_dir = output_dir / "trackers" / f"{benchmark}-{split}" / tracker_name / "data"
    if data_dir.parent.exists():
        shutil.rmtree(data_dir.parent)
    data_dir.mkdir(parents=True)

    source_dir = output_dir / "raw" / fold_name / "data"
    min_output_track_length = int(fold.get("min_output_track_length", 1))
    for sequence in fold_sequences(root, fold):
        source = source_dir / f"{sequence}.txt"
        require(source, f"{fold_name} raw output for {sequence}")
        rows = filter_short_tracks(read_mot(source), min_output_track_length)
        rows = link_tracklets(rows, fold["link"])
        write_mot(data_dir / f"{sequence}.txt", rows)
    return data_dir.parent


def run_w2c(root: Path, python: Path, config: dict, output_dir: Path, folds: list[str] | None) -> dict[str, Path]:
    for fold in config["folds"].values():
        runtime = merged_fold_runtime(config, fold)
        require(resolve_path(root, runtime["checkpoint_path"]), f"W2C checkpoint {runtime['checkpoint_path']}")
        require(resolve_path(root, runtime["reid_weights"]), f"W2C ReID weights {runtime['reid_weights']}")
        require(resolve_path(root, fold["det_dir"]), f"detector directory {fold['det_dir']}")

    fold_names = folds or list(config["folds"])
    unknown = [fold_name for fold_name in fold_names if fold_name not in config["folds"]]
    if unknown:
        raise ValueError(f"Unknown fold(s): {', '.join(unknown)}")

    gt_root = build_w2c_gt_view(root, config, output_dir)
    runtime_dir = output_dir / "runtime-configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for fold_name in fold_names:
        config_path = runtime_dir / f"{fold_name}.yaml"
        config_path.write_text(yaml.safe_dump(w2c_runtime(root, config, output_dir, gt_root, fold_name), sort_keys=False), encoding="utf-8")
        print(f"[run] {fold_name}")
        run_kraft(root, python, output_dir, fold_name, config_path)
        compose_w2c_fold(root, config, output_dir, fold_name)
        fold = config["folds"][fold_name]
        summaries[fold_name] = evaluate_trackeval(
            root=root,
            python=python,
            output_dir=output_dir,
            group_name=fold_name,
            tracker_name=f"{config['tracker_name']}-{fold_name}",
            gt_folder=gt_root / f"{fold['benchmark']}-{fold['split']}",
            trackers_folder=output_dir / "trackers",
            benchmark=fold["benchmark"],
            split=fold["split"],
            parallel=False,
        )
        print(f"[done] {fold_name}")
    return summaries


def compose_folded_result(root: Path, config: dict, output_dir: Path, fold_name: str) -> Path:
    fold = config["folds"][fold_name]
    benchmark = fold["benchmark"]
    split = fold["split"]
    tracker_name = f"{config['tracker_name']}-{fold_name}"
    data_dir = output_dir / "trackers" / f"{benchmark}-{split}" / tracker_name / "data"
    if data_dir.parent.exists():
        shutil.rmtree(data_dir.parent)
    data_dir.mkdir(parents=True, exist_ok=True)

    source_dir = output_dir / "raw" / fold_name / "data"
    min_output_track_length = int(fold.get("min_output_track_length", 1))
    for sequence in fold_sequences(root, fold):
        source = source_dir / f"{sequence}.txt"
        require(source, f"{fold_name} raw output for {sequence}")
        rows = filter_short_tracks(read_mot(source), min_output_track_length)
        rows = link_tracklets(rows, fold.get("link", {}))
        write_mot(data_dir / f"{sequence}.txt", rows)
    return data_dir.parent


def compose_packaged_tracks(root: Path, config: dict, output_dir: Path, fold_name: str) -> Path:
    fold = config["folds"][fold_name]
    benchmark = fold["benchmark"]
    split = fold["split"]
    tracker_name = f"{config['tracker_name']}-{fold_name}"
    data_dir = output_dir / "trackers" / f"{benchmark}-{split}" / tracker_name / "data"
    if data_dir.parent.exists():
        shutil.rmtree(data_dir.parent)
    data_dir.mkdir(parents=True, exist_ok=True)

    source_dir = resolve_path(root, fold["track_dir"])
    require(source_dir, f"{fold_name} packaged track directory")
    for sequence in fold_sequences(root, fold):
        source = source_dir / f"{sequence}.txt"
        require(source, f"{fold_name} packaged track for {sequence}")
        shutil.copy2(source, data_dir / f"{sequence}.txt")
    return data_dir.parent


def run_folded_mot(root: Path, python: Path, config: dict, output_dir: Path, folds: list[str] | None) -> dict[str, Path]:
    run_mode = config.get("run_mode", "track")
    if run_mode not in {"track", "evaluate_tracks"}:
        raise ValueError(f"Unsupported folded run_mode: {run_mode}")
    for fold in config["folds"].values():
        runtime = merged_fold_runtime(config, fold)
        if run_mode == "track":
            require(resolve_path(root, runtime["checkpoint_path"]), f"KRAfT checkpoint {runtime['checkpoint_path']}")
            require(resolve_path(root, runtime["reid_weights"]), f"ReID weights {runtime['reid_weights']}")
        require(resolve_path(root, fold["det_dir"]), f"detector directory {fold['det_dir']}")
        require(resolve_path(root, fold["source"]), f"fold source {fold['source']}")
        if run_mode == "evaluate_tracks":
            require(resolve_path(root, fold["track_dir"]), f"track directory {fold['track_dir']}")

    fold_names = folds or list(config["folds"])
    unknown = [fold_name for fold_name in fold_names if fold_name not in config["folds"]]
    if unknown:
        raise ValueError(f"Unknown fold(s): {', '.join(unknown)}")

    gt_root = build_mot_fold_gt_view(root, config, output_dir)
    runtime_dir = output_dir / "runtime-configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for fold_name in fold_names:
        config_path = runtime_dir / f"{fold_name}.yaml"
        print(f"[run] {fold_name}")
        if run_mode == "track":
            config_path.write_text(yaml.safe_dump(folded_runtime(root, config, output_dir, gt_root, fold_name), sort_keys=False), encoding="utf-8")
            run_kraft(root, python, output_dir, fold_name, config_path)
            compose_folded_result(root, config, output_dir, fold_name)
        else:
            compose_packaged_tracks(root, config, output_dir, fold_name)
        fold = config["folds"][fold_name]
        summaries[fold_name] = evaluate_trackeval(
            root=root,
            python=python,
            output_dir=output_dir,
            group_name=fold_name,
            tracker_name=f"{config['tracker_name']}-{fold_name}",
            gt_folder=gt_root / f"{fold['benchmark']}-{fold['split']}",
            trackers_folder=output_dir / "trackers",
            benchmark=fold["benchmark"],
            split=fold["split"],
            parallel=False,
        )
        print(f"[done] {fold_name}")
    return summaries


def evaluate_trackeval(
    root: Path,
    python: Path,
    output_dir: Path,
    group_name: str,
    tracker_name: str,
    gt_folder: Path,
    trackers_folder: Path,
    benchmark: str,
    split: str,
    parallel: bool,
) -> Path:
    script = root / "kraft" / "external" / "TrackEval" / "scripts" / "run_mot_challenge.py"
    require(script, "TrackEval run_mot_challenge.py")
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(script),
        "--USE_PARALLEL",
        "True" if parallel else "False",
        "--NUM_PARALLEL_CORES",
        "8" if parallel else "1",
        "--METRICS",
        "HOTA",
        "CLEAR",
        "Identity",
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--GT_FOLDER",
        str(gt_folder),
        "--TRACKERS_FOLDER",
        str(trackers_folder),
        "--BENCHMARK",
        benchmark,
        "--SPLIT_TO_EVAL",
        split,
        "--PLOT_CURVES",
        "False",
        "--PRINT_CONFIG",
        "False",
        "--TIME_PROGRESS",
        "False",
    ]
    with (output_dir / "logs" / f"{group_name}.trackeval.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    tracker_dir = trackers_folder / f"{benchmark}-{split}" / tracker_name
    for name in ("UAV_summary.txt", "pedestrian_summary.txt"):
        summary = tracker_dir / name
        if summary.exists():
            return summary
    raise FileNotFoundError(f"No TrackEval summary found in {tracker_dir}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result_name = config.get("result_name", config["dataset_type"])
    output_dir = args.output_dir.resolve() if args.output_dir else root / "outputs" / result_name
    output_dir.mkdir(parents=True, exist_ok=True)

    require(root / "kraft" / "main.py", "KRAfT main.py")
    require(resolve_path(root, config["dataset_root"]), f"{result_name} dataset root")

    if config["dataset_type"] == "uavswarm":
        summaries = run_uavswarm(root, args.python, config, output_dir, args.workers)
        include_overall = False
    elif config["dataset_type"] == "uavswarm-w2c":
        summaries = run_w2c(root, args.python, config, output_dir, args.folds)
        include_overall = args.folds is None or len(args.folds) > 1
    elif config["dataset_type"] == "muav":
        summaries = run_folded_mot(root, args.python, config, output_dir, args.folds)
        include_overall = args.folds is None or len(args.folds) > 1
    else:
        raise ValueError(f"Unsupported dataset_type: {config['dataset_type']}")

    write_results(root, config, summaries, include_overall=include_overall)
    print_results(config, summaries, include_overall=include_overall)


if __name__ == "__main__":
    main()
