from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import droplet_cv as dcv


@dataclass(frozen=True)
class EnsembleParams:
    same_frame_match_radius: float = 5.0
    temporal_support_radius: float = 8.0
    min_bgsub_p90: float = 27.0


@dataclass(frozen=True)
class HighRecallUnionParams:
    same_frame_match_radius: float = 3.0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iteration 2 ensemble: baseline + background-subtracted fusion."
    )
    parser.add_argument("--video", type=Path, default=dcv.DEFAULT_VIDEO_PATH)
    parser.add_argument("--annotations", type=Path, default=dcv.DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/iteration2_ensemble"),
    )
    return parser


def subset_annotations(
    annotations: dict[int, object],
    frame_ids: list[int],
) -> dict[int, object]:
    return {frame_id: annotations[frame_id] for frame_id in frame_ids}


def point_array(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty((0, 2), dtype=float)
    return df[["y", "x"]].to_numpy(dtype=float)


def min_dist(point: np.ndarray, points: np.ndarray) -> float:
    if len(points) == 0:
        return float("inf")
    return float(np.sqrt(np.sum((points - point) ** 2, axis=1)).min())


def build_branch_detections(
    raw_frames: np.ndarray,
    frame_ids: list[int],
    measurement_params: dcv.MeasurementParams,
    recovery_params: dcv.RecoveryParams,
    detector_params: dcv.DetectorParams,
    branch_name: str,
) -> pd.DataFrame:
    detections, _ = dcv.detect_video(
        raw_frames=raw_frames,
        detector_params=detector_params,
        measurement_params=measurement_params,
        frame_indices=frame_ids,
        branch_name=branch_name,
    )
    detections = dcv.recover_temporal_candidates(
        raw_frames=raw_frames,
        strong_detections=detections,
        detector_params=detector_params,
        measurement_params=measurement_params,
        recovery_params=recovery_params,
        branch_name=branch_name,
    )
    return detections[detections["frame"].isin(frame_ids)].reset_index(drop=True)


def ensemble_fuse_selective(
    baseline_detections: pd.DataFrame,
    bgsub_detections: pd.DataFrame,
    ensemble_params: EnsembleParams,
) -> pd.DataFrame:
    baseline_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in baseline_detections.groupby("frame")
    }
    bgsub_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in bgsub_detections.groupby("frame")
    }

    fused_rows = []
    frame_ids = sorted(
        set(baseline_by_frame.keys()).union(bgsub_by_frame.keys())
    )

    for frame_id in frame_ids:
        baseline_frame = baseline_by_frame.get(frame_id, pd.DataFrame())
        bgsub_frame = bgsub_by_frame.get(frame_id, pd.DataFrame())
        baseline_points = point_array(baseline_frame)
        prev_baseline = point_array(baseline_by_frame.get(frame_id - 1, pd.DataFrame()))
        next_baseline = point_array(baseline_by_frame.get(frame_id + 1, pd.DataFrame()))

        for _, row in baseline_frame.iterrows():
            fused = row.to_dict()
            fused["source_branch"] = "baseline"
            fused_rows.append(fused)

        for _, row in bgsub_frame.iterrows():
            point = np.array([float(row["y"]), float(row["x"])], dtype=float)
            if min_dist(point, baseline_points) <= ensemble_params.same_frame_match_radius:
                continue

            temporal_support = (
                min_dist(point, prev_baseline) <= ensemble_params.temporal_support_radius
                or min_dist(point, next_baseline) <= ensemble_params.temporal_support_radius
            )
            strong_enough = float(row["net_intensity_p90"]) >= ensemble_params.min_bgsub_p90

            if temporal_support and strong_enough:
                fused = row.to_dict()
                fused["source_branch"] = "bgsub_recovered"
                fused_rows.append(fused)

    if not fused_rows:
        return baseline_detections.copy()

    fused = pd.DataFrame(fused_rows).sort_values(["frame", "y", "x"]).reset_index(drop=True)
    return fused


def ensemble_fuse_high_recall_union(
    baseline_detections: pd.DataFrame,
    bgsub_detections: pd.DataFrame,
    union_params: HighRecallUnionParams,
) -> pd.DataFrame:
    baseline_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in baseline_detections.groupby("frame")
    }
    bgsub_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in bgsub_detections.groupby("frame")
    }

    fused_rows = []
    frame_ids = sorted(
        set(baseline_by_frame.keys()).union(bgsub_by_frame.keys())
    )

    for frame_id in frame_ids:
        baseline_frame = baseline_by_frame.get(frame_id, pd.DataFrame())
        bgsub_frame = bgsub_by_frame.get(frame_id, pd.DataFrame())
        baseline_points = point_array(baseline_frame)

        for _, row in baseline_frame.iterrows():
            fused = row.to_dict()
            fused["source_branch"] = "baseline"
            fused_rows.append(fused)

        for _, row in bgsub_frame.iterrows():
            point = np.array([float(row["y"]), float(row["x"])], dtype=float)
            if min_dist(point, baseline_points) <= union_params.same_frame_match_radius:
                continue
            fused = row.to_dict()
            fused["source_branch"] = "bgsub_union"
            fused_rows.append(fused)

    if not fused_rows:
        return baseline_detections.copy()

    fused = pd.DataFrame(fused_rows).sort_values(["frame", "y", "x"]).reset_index(drop=True)
    return fused


def evaluate_detector(
    detections: pd.DataFrame,
    annotations_subset: dict[int, object],
) -> tuple[dict[str, float], pd.DataFrame]:
    return dcv.evaluate_detections(
        detections=detections[["frame", "y", "x"]],
        annotations=annotations_subset,
        tolerance=6.0,
    )


def add_metric_row(rows: list[dict[str, object]], model_name: str, split_name: str, metrics: dict[str, float]) -> None:
    rows.append(
        {
            "model_name": model_name,
            "split": split_name,
            **metrics,
        }
    )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    video_path = dcv.resolve_project_path(args.video)
    annotations_path = dcv.resolve_project_path(args.annotations)
    output_dir = dcv.resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_frames, _ = dcv.load_video_grayscale(video_path)
    annotations = dcv.load_annotations(annotations_path)
    split_info = dcv.make_temporal_holdout_split(annotations, holdout_block_offset=1)
    train_annotations = subset_annotations(annotations, split_info["train_frames"])
    holdout_annotations = subset_annotations(annotations, split_info["holdout_frames"])

    measurement_params = dcv.MeasurementParams()
    recovery_params = dcv.RecoveryParams()
    baseline_params = dcv.DetectorParams(
        small_sigma=1.2,
        big_sigma=11.0,
        threshold_abs=20.5,
        min_distance=5,
        min_net_intensity_p90=31.0,
        min_net_intensity_mean=9.0,
    )
    bgsub_params = dcv.DetectorParams(
        small_sigma=1.2,
        big_sigma=9.0,
        threshold_abs=20.0,
        min_distance=5,
        min_net_intensity_p90=27.0,
        min_net_intensity_mean=9.0,
    )
    ensemble_params = EnsembleParams()
    high_recall_union_params = HighRecallUnionParams()

    all_frames = sorted(annotations)
    baseline_all = build_branch_detections(
        raw_frames, all_frames, measurement_params, recovery_params, baseline_params, "baseline_bandpass"
    )
    bgsub_all = build_branch_detections(
        raw_frames, all_frames, measurement_params, recovery_params, bgsub_params, "background_subtracted"
    )
    ensemble_all = ensemble_fuse_selective(baseline_all, bgsub_all, ensemble_params)
    ensemble_high_recall_all = ensemble_fuse_high_recall_union(
        baseline_all,
        bgsub_all,
        high_recall_union_params,
    )

    model_map = {
        "baseline_bandpass": baseline_all,
        "background_subtracted": bgsub_all,
        "ensemble_selective": ensemble_all,
        "ensemble_high_recall_union": ensemble_high_recall_all,
    }

    metrics_rows = []
    holdout_rows = []
    split_map = {
        "train": (split_info["train_frames"], train_annotations),
        "holdout": (split_info["holdout_frames"], holdout_annotations),
        "all_annotated": (all_frames, annotations),
    }

    for model_name, detections in model_map.items():
        for split_name, (frame_ids, annotations_subset) in split_map.items():
            subset = detections[detections["frame"].isin(frame_ids)].reset_index(drop=True)
            metrics, per_frame = evaluate_detector(subset, annotations_subset)
            add_metric_row(metrics_rows, model_name, split_name, metrics)
            if split_name == "holdout":
                tmp = per_frame.copy()
                tmp["model_name"] = model_name
                tmp["error_score"] = tmp["fp"] + tmp["fn"]
                holdout_rows.append(tmp)

    metrics_df = pd.DataFrame(metrics_rows)
    holdout_per_frame_df = pd.concat(holdout_rows, ignore_index=True)

    metrics_df.to_csv(output_dir / "ensemble_metrics.csv", index=False)
    holdout_per_frame_df.to_csv(output_dir / "holdout_per_frame.csv", index=False)
    ensemble_all.to_csv(output_dir / "ensemble_selective_detections.csv", index=False)
    ensemble_high_recall_all.to_csv(output_dir / "ensemble_high_recall_detections.csv", index=False)

    summary = {
        "split": {
            "blocks": split_info["blocks"],
            "train_frames": split_info["train_frames"],
            "holdout_frames": split_info["holdout_frames"],
        },
        "baseline_params": asdict(baseline_params),
        "background_subtracted_params": asdict(bgsub_params),
        "ensemble_selective_params": asdict(ensemble_params),
        "ensemble_high_recall_union_params": asdict(high_recall_union_params),
    }
    with open(output_dir / "ensemble_config.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    holdout = metrics_df[metrics_df["split"] == "holdout"].copy().sort_values(
        ["f1", "precision", "recall"], ascending=[False, False, False]
    )
    lines = [
        "# Iteration 2 Ensemble",
        "",
        "## Holdout results",
        "",
        "| Model | Precision | Recall | F1 | Count MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in holdout.iterrows():
        lines.append(
            f"| {row['model_name']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['count_mae']:.2f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Iteration 2 ensemble saved to:", output_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
