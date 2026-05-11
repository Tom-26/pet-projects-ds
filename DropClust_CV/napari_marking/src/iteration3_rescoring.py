from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import droplet_cv as dcv
from iteration2_ensemble import build_branch_detections, point_array, min_dist, subset_annotations


@dataclass(frozen=True)
class RescoringParams:
    merge_radius: float = 3.0
    support_radius: float = 8.0
    positive_tol: float = 4.0
    negative_tol: float = 7.0
    positive_weight: float = 2.0
    l2_reg: float = 1e-2
    learning_rate: float = 0.05
    num_steps: int = 2500


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iteration 3 candidate rescoring on merged detector candidates."
    )
    parser.add_argument("--video", type=Path, default=dcv.DEFAULT_VIDEO_PATH)
    parser.add_argument("--annotations", type=Path, default=dcv.DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/iteration3_rescoring"),
    )
    return parser


def baseline_params() -> dcv.DetectorParams:
    return dcv.DetectorParams(
        small_sigma=1.2,
        big_sigma=11.0,
        threshold_abs=20.5,
        min_distance=5,
        min_net_intensity_p90=31.0,
        min_net_intensity_mean=9.0,
    )


def bgsub_params() -> dcv.DetectorParams:
    return dcv.DetectorParams(
        small_sigma=1.2,
        big_sigma=9.0,
        threshold_abs=20.0,
        min_distance=5,
        min_net_intensity_p90=27.0,
        min_net_intensity_mean=9.0,
    )


def merge_branch_candidates(
    baseline_detections: pd.DataFrame,
    bgsub_detections: pd.DataFrame,
    merge_radius: float,
) -> pd.DataFrame:
    rows = []
    baseline_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in baseline_detections.groupby("frame")
    }
    bgsub_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in bgsub_detections.groupby("frame")
    }

    frame_ids = sorted(set(baseline_by_frame).union(bgsub_by_frame))
    for frame_id in frame_ids:
        candidates: list[dict[str, float | int]] = []
        assigned_bgsub = set()
        baseline_frame = baseline_by_frame.get(frame_id, pd.DataFrame())
        bgsub_frame = bgsub_by_frame.get(frame_id, pd.DataFrame())

        for _, row in baseline_frame.iterrows():
            candidate = row.to_dict()
            candidate["source_baseline"] = 1
            candidate["source_bgsub"] = 0
            candidate["branch_agreement"] = 0
            candidate["bgsub_filtered_peak"] = 0.0
            candidate["bgsub_net_intensity_p90"] = 0.0
            candidate["bgsub_net_intensity_mean"] = 0.0
            candidate["base_filtered_peak"] = float(row["filtered_peak"])
            candidate["base_net_intensity_p90"] = float(row["net_intensity_p90"])
            candidate["base_net_intensity_mean"] = float(row["net_intensity_mean"])

            best_idx = None
            best_dist = float("inf")
            for bg_idx, bg_row in bgsub_frame.iterrows():
                if bg_idx in assigned_bgsub:
                    continue
                dist = float(np.hypot(float(row["y"]) - float(bg_row["y"]), float(row["x"]) - float(bg_row["x"])))
                if dist <= merge_radius and dist < best_dist:
                    best_dist = dist
                    best_idx = bg_idx

            if best_idx is not None:
                assigned_bgsub.add(best_idx)
                bg_row = bgsub_frame.loc[best_idx]
                candidate["source_bgsub"] = 1
                candidate["branch_agreement"] = 1
                candidate["bgsub_filtered_peak"] = float(bg_row["filtered_peak"])
                candidate["bgsub_net_intensity_p90"] = float(bg_row["net_intensity_p90"])
                candidate["bgsub_net_intensity_mean"] = float(bg_row["net_intensity_mean"])
                if float(bg_row["filtered_peak"]) > float(row["filtered_peak"]):
                    candidate["y"] = float(bg_row["y"])
                    candidate["x"] = float(bg_row["x"])
                    candidate["filtered_peak"] = float(bg_row["filtered_peak"])
                    candidate["net_intensity_mean"] = float(bg_row["net_intensity_mean"])
                    candidate["net_intensity_peak"] = float(bg_row["net_intensity_peak"])
                    candidate["net_intensity_p90"] = float(bg_row["net_intensity_p90"])
                    candidate["raw_signal_mean"] = float(bg_row["raw_signal_mean"])
                    candidate["raw_signal_peak"] = float(bg_row["raw_signal_peak"])
                    candidate["raw_signal_p90"] = float(bg_row["raw_signal_p90"])
                    candidate["raw_background_median"] = float(bg_row["raw_background_median"])

            candidates.append(candidate)

        for bg_idx, bg_row in bgsub_frame.iterrows():
            if bg_idx in assigned_bgsub:
                continue
            candidate = bg_row.to_dict()
            candidate["source_baseline"] = 0
            candidate["source_bgsub"] = 1
            candidate["branch_agreement"] = 0
            candidate["base_filtered_peak"] = 0.0
            candidate["base_net_intensity_p90"] = 0.0
            candidate["base_net_intensity_mean"] = 0.0
            candidate["bgsub_filtered_peak"] = float(bg_row["filtered_peak"])
            candidate["bgsub_net_intensity_p90"] = float(bg_row["net_intensity_p90"])
            candidate["bgsub_net_intensity_mean"] = float(bg_row["net_intensity_mean"])
            candidates.append(candidate)

        for idx, candidate in enumerate(candidates):
            candidate["candidate_id"] = int(idx)
            candidate["frame"] = int(frame_id)
            rows.append(candidate)

    return pd.DataFrame(rows).sort_values(["frame", "y", "x"]).reset_index(drop=True)


def add_temporal_support_features(candidates: pd.DataFrame, support_radius: float) -> pd.DataFrame:
    enriched = candidates.copy()
    by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in enriched.groupby("frame")
    }

    base_by_frame = {
        int(frame_id): point_array(group[group["source_baseline"] == 1])
        for frame_id, group in by_frame.items()
    }
    bg_by_frame = {
        int(frame_id): point_array(group[group["source_bgsub"] == 1])
        for frame_id, group in by_frame.items()
    }

    rows = []
    for frame_id, frame_group in by_frame.items():
        prev_base = base_by_frame.get(frame_id - 1, np.empty((0, 2)))
        next_base = base_by_frame.get(frame_id + 1, np.empty((0, 2)))
        prev_bg = bg_by_frame.get(frame_id - 1, np.empty((0, 2)))
        next_bg = bg_by_frame.get(frame_id + 1, np.empty((0, 2)))

        for _, row in frame_group.iterrows():
            point = np.array([float(row["y"]), float(row["x"])], dtype=float)
            row = row.to_dict()

            row["base_prev_support"] = float(min_dist(point, prev_base) <= support_radius)
            row["base_next_support"] = float(min_dist(point, next_base) <= support_radius)
            row["bg_prev_support"] = float(min_dist(point, prev_bg) <= support_radius)
            row["bg_next_support"] = float(min_dist(point, next_bg) <= support_radius)
            row["temporal_support_sum"] = (
                row["base_prev_support"]
                + row["base_next_support"]
                + row["bg_prev_support"]
                + row["bg_next_support"]
            )
            row["temporal_support_any"] = float(row["temporal_support_sum"] > 0)
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["frame", "y", "x"]).reset_index(drop=True)


def assign_training_labels(
    candidates: pd.DataFrame,
    annotations: dict[int, np.ndarray],
    positive_tol: float,
    negative_tol: float,
) -> pd.DataFrame:
    rows = []
    for frame_id, frame_group in candidates.groupby("frame"):
        truth = annotations.get(int(frame_id), np.empty((0, 2), dtype=float))
        truth = np.asarray(truth, dtype=float)

        for _, row in frame_group.iterrows():
            point = np.array([float(row["y"]), float(row["x"])], dtype=float)
            if len(truth) == 0:
                min_truth_dist = float("inf")
            else:
                min_truth_dist = float(np.sqrt(np.sum((truth - point) ** 2, axis=1)).min())

            label = None
            if min_truth_dist <= positive_tol:
                label = 1
            elif min_truth_dist >= negative_tol:
                label = 0

            row = row.to_dict()
            row["min_truth_distance"] = min_truth_dist
            row["label"] = label
            rows.append(row)

    labeled = pd.DataFrame(rows)
    return labeled


FEATURE_COLUMNS = [
    "filtered_peak",
    "net_intensity_mean",
    "net_intensity_peak",
    "net_intensity_p90",
    "raw_signal_peak",
    "raw_signal_mean",
    "raw_background_median",
    "source_baseline",
    "source_bgsub",
    "branch_agreement",
    "base_filtered_peak",
    "base_net_intensity_p90",
    "base_net_intensity_mean",
    "bgsub_filtered_peak",
    "bgsub_net_intensity_p90",
    "bgsub_net_intensity_mean",
    "base_prev_support",
    "base_next_support",
    "bg_prev_support",
    "bg_next_support",
    "temporal_support_sum",
    "temporal_support_any",
]


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    params: RescoringParams,
) -> np.ndarray:
    x_aug = np.hstack([x, np.ones((len(x), 1), dtype=float)])
    weights = np.zeros(x_aug.shape[1], dtype=float)
    sample_weights = np.where(y > 0.5, params.positive_weight, 1.0)

    for _ in range(params.num_steps):
        logits = x_aug @ weights
        probs = expit(logits)
        error = (probs - y) * sample_weights
        grad = (x_aug.T @ error) / len(x_aug)
        grad[:-1] += params.l2_reg * weights[:-1]
        weights -= params.learning_rate * grad

    return weights


def predict_scores(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([x, np.ones((len(x), 1), dtype=float)])
    return expit(x_aug @ weights)


def select_threshold(
    labeled_train: pd.DataFrame,
    score_column: str,
) -> float:
    best_threshold = 0.5
    best_score = -1.0
    candidates = np.linspace(0.20, 0.90, 71)

    y_true = labeled_train["label"].to_numpy(dtype=float)
    scores = labeled_train[score_column].to_numpy(dtype=float)

    for threshold in candidates:
        y_pred = scores >= threshold
        tp = float(np.sum((y_true == 1) & y_pred))
        fp = float(np.sum((y_true == 0) & y_pred))
        fn = float(np.sum((y_true == 1) & (~y_pred)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # Small recall preference to keep balance from collapsing.
        score = f1 + 0.02 * recall
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def filter_by_threshold(
    candidates: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    kept = candidates[candidates[score_column] >= threshold].copy()
    return kept.sort_values(["frame", "y", "x"]).reset_index(drop=True)


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
    params = RescoringParams()

    all_frames = sorted(annotations)
    baseline_all = build_branch_detections(
        raw_frames, all_frames, measurement_params, recovery_params, baseline_params(), "baseline_bandpass"
    )
    bgsub_all = build_branch_detections(
        raw_frames, all_frames, measurement_params, recovery_params, bgsub_params(), "background_subtracted"
    )

    candidates = merge_branch_candidates(
        baseline_detections=baseline_all,
        bgsub_detections=bgsub_all,
        merge_radius=params.merge_radius,
    )
    candidates = add_temporal_support_features(candidates, support_radius=params.support_radius)

    train_candidates = candidates[candidates["frame"].isin(split_info["train_frames"])].reset_index(drop=True)
    holdout_candidates = candidates[candidates["frame"].isin(split_info["holdout_frames"])].reset_index(drop=True)
    all_candidates = candidates.copy()

    train_labeled = assign_training_labels(
        train_candidates,
        train_annotations,
        positive_tol=params.positive_tol,
        negative_tol=params.negative_tol,
    )
    train_fit = train_labeled.dropna(subset=["label"]).reset_index(drop=True)

    train_x = train_fit[FEATURE_COLUMNS].to_numpy(dtype=float)
    train_y = train_fit["label"].to_numpy(dtype=float)
    holdout_x = holdout_candidates[FEATURE_COLUMNS].to_numpy(dtype=float)
    all_x = all_candidates[FEATURE_COLUMNS].to_numpy(dtype=float)

    train_x_std, holdout_x_std, mean, std = standardize(train_x, holdout_x)
    _, all_x_std, _, _ = standardize(train_x, all_x)

    weights = fit_logistic_regression(train_x_std, train_y, params)

    train_fit = train_fit.copy()
    train_fit["rescoring_score"] = predict_scores(train_x_std, weights)
    threshold = select_threshold(train_fit, "rescoring_score")

    holdout_candidates = holdout_candidates.copy()
    holdout_candidates["rescoring_score"] = predict_scores(holdout_x_std, weights)
    all_candidates = all_candidates.copy()
    all_candidates["rescoring_score"] = predict_scores(all_x_std, weights)

    baseline_holdout = baseline_all[baseline_all["frame"].isin(split_info["holdout_frames"])].reset_index(drop=True)
    bgsub_holdout = bgsub_all[bgsub_all["frame"].isin(split_info["holdout_frames"])].reset_index(drop=True)
    rescored_holdout = filter_by_threshold(holdout_candidates, "rescoring_score", threshold)

    metrics_rows = []
    for model_name, detections, ann_subset in [
        ("baseline_bandpass", baseline_holdout, holdout_annotations),
        ("background_subtracted", bgsub_holdout, holdout_annotations),
        ("rescored_union", rescored_holdout, holdout_annotations),
    ]:
        metrics, _ = dcv.evaluate_detections(
            detections=detections[["frame", "y", "x"]],
            annotations=ann_subset,
            tolerance=6.0,
        )
        metrics_rows.append({"model_name": model_name, "split": "holdout", **metrics})

    rescored_all = filter_by_threshold(all_candidates, "rescoring_score", threshold)
    all_metrics, all_pf = dcv.evaluate_detections(
        detections=rescored_all[["frame", "y", "x"]],
        annotations=annotations,
        tolerance=6.0,
    )
    metrics_rows.append({"model_name": "rescored_union", "split": "all_annotated", **all_metrics})

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "rescoring_metrics.csv", index=False)
    rescored_all.to_csv(output_dir / "rescored_detections.csv", index=False)
    all_pf.to_csv(output_dir / "rescored_all_per_frame.csv", index=False)
    train_fit.to_csv(output_dir / "train_labeled_candidates.csv", index=False)
    holdout_candidates.to_csv(output_dir / "holdout_candidates_scored.csv", index=False)

    model_summary = {
        "split": {
            "blocks": split_info["blocks"],
            "train_frames": split_info["train_frames"],
            "holdout_frames": split_info["holdout_frames"],
        },
        "baseline_params": asdict(baseline_params()),
        "background_subtracted_params": asdict(bgsub_params()),
        "rescoring_params": asdict(params),
        "feature_columns": FEATURE_COLUMNS,
        "score_threshold": threshold,
        "weights": weights.tolist(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "train_candidate_count": int(len(train_candidates)),
        "train_fit_count": int(len(train_fit)),
        "holdout_candidate_count": int(len(holdout_candidates)),
    }
    with open(output_dir / "rescoring_config.json", "w", encoding="utf-8") as fh:
        json.dump(model_summary, fh, ensure_ascii=False, indent=2)

    lines = [
        "# Iteration 3 Rescoring",
        "",
        f"- selected threshold: `{threshold:.3f}`",
        f"- train fitted candidates: `{len(train_fit)}`",
        f"- holdout candidates scored: `{len(holdout_candidates)}`",
        "",
        "## Holdout results",
        "",
        "| Model | Precision | Recall | F1 | Count MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics_df[metrics_df["split"] == "holdout"].iterrows():
        lines.append(
            f"| {row['model_name']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['count_mae']:.2f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Iteration 3 rescoring saved to:", output_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
