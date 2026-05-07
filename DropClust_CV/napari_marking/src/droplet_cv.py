from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = Path("data/raw/input.avi")
DEFAULT_ANNOTATIONS_PATH = Path("data/annotations/manual_points.csv")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


@dataclass(frozen=True)
class DetectorParams:
    small_sigma: float = 1.2
    big_sigma: float = 11.0
    threshold_abs: float = 20.5
    min_distance: int = 5
    min_net_intensity_p90: float = 29.0
    min_net_intensity_mean: float = 11.0


@dataclass(frozen=True)
class MeasurementParams:
    signal_radius: int = 4
    background_radius: int = 9


@dataclass(frozen=True)
class TrackingParams:
    max_distance: float = 10.0
    max_gap: int = 3
    min_track_length: int = 1


@dataclass(frozen=True)
class RecoveryParams:
    enabled: bool = True
    threshold_margin: float = 1.0
    p90_margin: float = 5.0
    mean_margin: float = 3.0
    same_frame_radius: float = 2.0
    temporal_radius: float = 10.0


def load_video_grayscale(video_path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames were read from: {video_path}")

    return np.stack(frames, axis=0), float(fps)


def load_annotations(csv_path: Path) -> dict[int, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    grouped = {}
    for frame_id, group in df.groupby("frame"):
        grouped[int(frame_id)] = group[["y", "x"]].to_numpy(dtype=float)
    return grouped


def bandpass_filter_video(
    frames: np.ndarray,
    small_sigma: float,
    big_sigma: float,
) -> np.ndarray:
    frames_float = frames.astype(np.float32)
    small_blur = gaussian_filter(frames_float, sigma=(0, small_sigma, small_sigma))
    big_blur = gaussian_filter(frames_float, sigma=(0, big_sigma, big_sigma))
    return np.clip(small_blur - big_blur, a_min=0.0, a_max=None)


def detect_frame_peaks(filtered_frame: np.ndarray, params: DetectorParams) -> np.ndarray:
    coords = peak_local_max(
        filtered_frame,
        min_distance=params.min_distance,
        threshold_abs=params.threshold_abs,
        exclude_border=False,
    )
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float)
    return coords.astype(float)


def apply_feature_filters(
    detections: pd.DataFrame,
    params: DetectorParams,
) -> pd.DataFrame:
    if detections.empty:
        return detections.copy()

    keep_mask = (
        (detections["net_intensity_p90"] >= params.min_net_intensity_p90)
        & (detections["net_intensity_mean"] >= params.min_net_intensity_mean)
    )
    return detections[keep_mask].reset_index(drop=True)


def extract_patch(
    image: np.ndarray,
    center_y: float,
    center_x: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0 = max(int(np.floor(center_y)) - radius, 0)
    y1 = min(int(np.floor(center_y)) + radius + 1, image.shape[0])
    x0 = max(int(np.floor(center_x)) - radius, 0)
    x1 = min(int(np.floor(center_x)) + radius + 1, image.shape[1])

    patch = image[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    return patch, yy, xx


def measure_detection(
    raw_frame: np.ndarray,
    filtered_frame: np.ndarray,
    y: float,
    x: float,
    params: MeasurementParams,
) -> dict[str, float]:
    patch, yy, xx = extract_patch(raw_frame, y, x, params.background_radius)
    distances = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)

    signal_mask = distances <= params.signal_radius
    background_mask = (distances > params.signal_radius) & (
        distances <= params.background_radius
    )

    signal_values = patch[signal_mask]
    if signal_values.size == 0:
        signal_values = np.array([raw_frame[int(round(y)), int(round(x))]], dtype=float)

    if np.any(background_mask):
        background_values = patch[background_mask]
        background_median = float(np.median(background_values))
    else:
        background_median = float(np.median(patch))

    iy = int(np.clip(round(y), 0, raw_frame.shape[0] - 1))
    ix = int(np.clip(round(x), 0, raw_frame.shape[1] - 1))

    signal_mean = float(np.mean(signal_values))
    signal_peak = float(np.max(signal_values))
    signal_p90 = float(np.percentile(signal_values, 90))
    filtered_peak = float(filtered_frame[iy, ix])

    return {
        "y": float(y),
        "x": float(x),
        "raw_signal_mean": signal_mean,
        "raw_signal_peak": signal_peak,
        "raw_signal_p90": signal_p90,
        "raw_background_median": background_median,
        "net_intensity_mean": signal_mean - background_median,
        "net_intensity_peak": signal_peak - background_median,
        "net_intensity_p90": signal_p90 - background_median,
        "filtered_peak": filtered_peak,
    }


def detect_video(
    raw_frames: np.ndarray,
    detector_params: DetectorParams,
    measurement_params: MeasurementParams,
    frame_indices: list[int] | None = None,
    apply_filters: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    filtered_frames = bandpass_filter_video(
        raw_frames,
        small_sigma=detector_params.small_sigma,
        big_sigma=detector_params.big_sigma,
    )

    target_frames = frame_indices if frame_indices is not None else list(range(len(raw_frames)))
    rows: list[dict[str, float | int]] = []

    for frame_id in target_frames:
        coords = detect_frame_peaks(filtered_frames[frame_id], detector_params)
        for y, x in coords:
            measurement = measure_detection(
                raw_frame=raw_frames[frame_id],
                filtered_frame=filtered_frames[frame_id],
                y=y,
                x=x,
                params=measurement_params,
            )
            measurement["frame"] = int(frame_id)
            rows.append(measurement)

    detections = pd.DataFrame(rows)
    if detections.empty:
        detections = pd.DataFrame(
            columns=[
                "frame",
                "y",
                "x",
                "raw_signal_mean",
                "raw_signal_peak",
                "raw_signal_p90",
                "raw_background_median",
                "net_intensity_mean",
                "net_intensity_peak",
                "net_intensity_p90",
                "filtered_peak",
            ]
        )

    if apply_filters:
        detections = apply_feature_filters(detections, detector_params)

    return detections, filtered_frames


def recover_temporal_candidates(
    raw_frames: np.ndarray,
    strong_detections: pd.DataFrame,
    detector_params: DetectorParams,
    measurement_params: MeasurementParams,
    recovery_params: RecoveryParams,
) -> pd.DataFrame:
    if not recovery_params.enabled:
        return strong_detections

    weak_params = DetectorParams(
        small_sigma=detector_params.small_sigma,
        big_sigma=detector_params.big_sigma,
        threshold_abs=max(0.0, detector_params.threshold_abs - recovery_params.threshold_margin),
        min_distance=detector_params.min_distance,
        min_net_intensity_p90=max(
            0.0, detector_params.min_net_intensity_p90 - recovery_params.p90_margin
        ),
        min_net_intensity_mean=detector_params.min_net_intensity_mean - recovery_params.mean_margin,
    )
    weak_detections, _ = detect_video(
        raw_frames=raw_frames,
        detector_params=weak_params,
        measurement_params=measurement_params,
    )

    strong_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in strong_detections.groupby("frame")
    }
    weak_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in weak_detections.groupby("frame")
    }

    recovered_rows = []
    for frame_id in range(len(raw_frames)):
        strong_frame = strong_by_frame.get(frame_id, pd.DataFrame())
        weak_frame = weak_by_frame.get(frame_id, pd.DataFrame())
        if weak_frame.empty:
            continue

        strong_points = (
            strong_frame[["y", "x"]].to_numpy(dtype=float)
            if not strong_frame.empty
            else np.empty((0, 2), dtype=float)
        )
        prev_frame = strong_by_frame.get(frame_id - 1, pd.DataFrame())
        next_frame = strong_by_frame.get(frame_id + 1, pd.DataFrame())
        prev_points = (
            prev_frame[["y", "x"]].to_numpy(dtype=float)
            if not prev_frame.empty
            else np.empty((0, 2), dtype=float)
        )
        next_points = (
            next_frame[["y", "x"]].to_numpy(dtype=float)
            if not next_frame.empty
            else np.empty((0, 2), dtype=float)
        )

        for _, row in weak_frame.iterrows():
            point = np.array([float(row["y"]), float(row["x"])], dtype=float)
            if len(strong_points):
                same_frame_dist = np.sqrt(np.sum((strong_points - point) ** 2, axis=1))
                if float(np.min(same_frame_dist)) <= recovery_params.same_frame_radius:
                    continue

            has_prev_support = False
            has_next_support = False
            if len(prev_points):
                prev_dist = np.sqrt(np.sum((prev_points - point) ** 2, axis=1))
                has_prev_support = float(np.min(prev_dist)) <= recovery_params.temporal_radius
            if len(next_points):
                next_dist = np.sqrt(np.sum((next_points - point) ** 2, axis=1))
                has_next_support = float(np.min(next_dist)) <= recovery_params.temporal_radius

            if has_prev_support or has_next_support:
                recovered_rows.append(row.to_dict())

    if not recovered_rows:
        return strong_detections.sort_values(["frame", "y", "x"]).reset_index(drop=True)

    recovered = pd.DataFrame(recovered_rows)
    merged = pd.concat([strong_detections, recovered], ignore_index=True)
    merged = merged.sort_values(["frame", "y", "x"]).reset_index(drop=True)
    return merged


def match_points(
    predicted: np.ndarray,
    truth: np.ndarray,
    tolerance: float,
) -> tuple[int, int, int]:
    if len(predicted) == 0 and len(truth) == 0:
        return 0, 0, 0
    if len(predicted) == 0:
        return 0, 0, len(truth)
    if len(truth) == 0:
        return 0, len(predicted), 0

    distances = cdist(predicted, truth)
    row_ind, col_ind = linear_sum_assignment(distances)

    true_positive = 0
    for row, col in zip(row_ind, col_ind):
        if distances[row, col] <= tolerance:
            true_positive += 1

    false_positive = len(predicted) - true_positive
    false_negative = len(truth) - true_positive
    return true_positive, false_positive, false_negative


def match_points_detail(
    predicted: np.ndarray,
    truth: np.ndarray,
    tolerance: float,
) -> dict[str, np.ndarray]:
    predicted = np.asarray(predicted, dtype=float)
    truth = np.asarray(truth, dtype=float)

    if len(predicted) == 0:
        return {
            "tp_pred_idx": np.empty(0, dtype=int),
            "fp_pred_idx": np.empty(0, dtype=int),
            "matched_truth_idx": np.empty(0, dtype=int),
            "fn_truth_idx": np.arange(len(truth), dtype=int),
        }
    if len(truth) == 0:
        return {
            "tp_pred_idx": np.empty(0, dtype=int),
            "fp_pred_idx": np.arange(len(predicted), dtype=int),
            "matched_truth_idx": np.empty(0, dtype=int),
            "fn_truth_idx": np.empty(0, dtype=int),
        }

    distances = cdist(predicted, truth)
    row_ind, col_ind = linear_sum_assignment(distances)

    matched_pred = []
    matched_truth = []
    for row, col in zip(row_ind, col_ind):
        if distances[row, col] <= tolerance:
            matched_pred.append(int(row))
            matched_truth.append(int(col))

    matched_pred_idx = np.array(sorted(matched_pred), dtype=int)
    matched_truth_idx = np.array(sorted(matched_truth), dtype=int)
    fp_pred_idx = np.setdiff1d(np.arange(len(predicted), dtype=int), matched_pred_idx)
    fn_truth_idx = np.setdiff1d(np.arange(len(truth), dtype=int), matched_truth_idx)

    return {
        "tp_pred_idx": matched_pred_idx,
        "fp_pred_idx": fp_pred_idx,
        "matched_truth_idx": matched_truth_idx,
        "fn_truth_idx": fn_truth_idx,
    }


def evaluate_detections(
    detections: pd.DataFrame,
    annotations: dict[int, np.ndarray],
    tolerance: float = 6.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    predicted_by_frame = {
        int(frame_id): group[["y", "x"]].to_numpy(dtype=float)
        for frame_id, group in detections.groupby("frame")
    }

    per_frame_rows = []
    total_tp = total_fp = total_fn = 0
    count_errors = []

    for frame_id in sorted(annotations):
        predicted = predicted_by_frame.get(frame_id, np.empty((0, 2), dtype=float))
        truth = annotations[frame_id]
        tp, fp, fn = match_points(predicted, truth, tolerance=tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        count_errors.append(abs(len(predicted) - len(truth)))
        per_frame_rows.append(
            {
                "frame": frame_id,
                "predicted_count": int(len(predicted)),
                "truth_count": int(len(truth)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "count_mae": float(np.mean(count_errors)) if count_errors else 0.0,
        "annotated_frames": int(len(annotations)),
    }
    return metrics, pd.DataFrame(per_frame_rows)


def parse_grid(text: str, cast) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def tune_detector(
    raw_frames: np.ndarray,
    annotations: dict[int, np.ndarray],
    measurement_params: MeasurementParams,
    small_sigmas: list[float],
    big_sigmas: list[float],
    thresholds: list[float],
    min_distances: list[int],
    net_p90_thresholds: list[float],
    net_mean_thresholds: list[float],
    tolerance: float,
) -> tuple[DetectorParams, pd.DataFrame]:
    frame_indices = sorted(annotations)
    if not frame_indices:
        raise ValueError("No annotation frames were provided for tuning.")

    candidate_cache: dict[tuple[float, float, float, int], pd.DataFrame] = {}
    result_rows = []

    for small_sigma, big_sigma, threshold_abs, min_distance in product(
        small_sigmas,
        big_sigmas,
        thresholds,
        min_distances,
    ):
        spatial_params = DetectorParams(
            small_sigma=small_sigma,
            big_sigma=big_sigma,
            threshold_abs=threshold_abs,
            min_distance=min_distance,
            min_net_intensity_p90=0.0,
            min_net_intensity_mean=-999.0,
        )
        candidates, _ = detect_video(
            raw_frames=raw_frames,
            detector_params=spatial_params,
            measurement_params=measurement_params,
            frame_indices=frame_indices,
            apply_filters=False,
        )
        candidate_cache[(small_sigma, big_sigma, threshold_abs, min_distance)] = candidates

    for small_sigma, big_sigma, threshold_abs, min_distance, p90_threshold, mean_threshold in product(
        small_sigmas,
        big_sigmas,
        thresholds,
        min_distances,
        net_p90_thresholds,
        net_mean_thresholds,
    ):
        candidates = candidate_cache[(small_sigma, big_sigma, threshold_abs, min_distance)]
        filtered = candidates[
            (candidates["net_intensity_p90"] >= p90_threshold)
            & (candidates["net_intensity_mean"] >= mean_threshold)
        ]
        metrics, _ = evaluate_detections(
            detections=filtered[["frame", "y", "x"]],
            annotations=annotations,
            tolerance=tolerance,
        )

        result_rows.append(
            {
                "small_sigma": small_sigma,
                "big_sigma": big_sigma,
                "threshold_abs": threshold_abs,
                "min_distance": min_distance,
                "min_net_intensity_p90": p90_threshold,
                "min_net_intensity_mean": mean_threshold,
                **metrics,
            }
        )

    results = pd.DataFrame(result_rows).sort_values(
        by=["f1", "precision", "recall", "count_mae"],
        ascending=[False, False, False, True],
    )
    best_row = results.iloc[0]
    best_params = DetectorParams(
        small_sigma=float(best_row["small_sigma"]),
        big_sigma=float(best_row["big_sigma"]),
        threshold_abs=float(best_row["threshold_abs"]),
        min_distance=int(best_row["min_distance"]),
    )
    return best_params, results


def select_best_params(
    results: pd.DataFrame,
    selection_metric: str,
) -> tuple[DetectorParams, pd.DataFrame]:
    ranked = results.copy()

    if selection_metric == "f1":
        ranked = ranked.sort_values(
            by=["f1", "precision", "recall", "count_mae"],
            ascending=[False, False, False, True],
        )
    elif selection_metric == "count_mae":
        ranked = ranked.sort_values(
            by=["count_mae", "f1", "precision", "recall"],
            ascending=[True, False, False, False],
        )
    else:
        # Mild penalty on count error keeps F1 dominant while preferring saner counts.
        ranked["balanced_score"] = ranked["f1"] - 0.001 * ranked["count_mae"]
        ranked = ranked.sort_values(
            by=["balanced_score", "f1", "precision", "recall"],
            ascending=[False, False, False, False],
        )

    best_row = ranked.iloc[0]
    best_params = DetectorParams(
        small_sigma=float(best_row["small_sigma"]),
        big_sigma=float(best_row["big_sigma"]),
        threshold_abs=float(best_row["threshold_abs"]),
        min_distance=int(best_row["min_distance"]),
        min_net_intensity_p90=float(best_row["min_net_intensity_p90"]),
        min_net_intensity_mean=float(best_row["min_net_intensity_mean"]),
    )
    return best_params, ranked


def track_detections(
    detections: pd.DataFrame,
    tracking_params: TrackingParams,
) -> pd.DataFrame:
    if detections.empty:
        detections = detections.copy()
        detections["track_id"] = pd.Series(dtype=int)
        return detections

    tracked = detections.sort_values(["frame", "y", "x"]).reset_index(drop=True).copy()
    tracked["track_id"] = -1

    next_track_id = 0
    active_tracks: dict[int, dict[str, float]] = {}

    for frame_id, frame_group in tracked.groupby("frame", sort=True):
        frame_id = int(frame_id)
        active_tracks = {
            track_id: state
            for track_id, state in active_tracks.items()
            if frame_id - int(state["last_frame"]) <= tracking_params.max_gap + 1
        }

        det_indices = frame_group.index.to_list()
        det_points = tracked.loc[det_indices, ["y", "x"]].to_numpy(dtype=float)
        matched_detections: set[int] = set()

        if active_tracks and len(det_points) > 0:
            active_ids = list(active_tracks)
            active_points = np.array(
                [[active_tracks[track_id]["y"], active_tracks[track_id]["x"]] for track_id in active_ids],
                dtype=float,
            )
            distances = cdist(active_points, det_points)
            row_ind, col_ind = linear_sum_assignment(distances)

            for row, col in zip(row_ind, col_ind):
                if distances[row, col] > tracking_params.max_distance:
                    continue

                track_id = active_ids[row]
                det_index = det_indices[col]
                tracked.at[det_index, "track_id"] = track_id
                active_tracks[track_id] = {
                    "last_frame": float(frame_id),
                    "y": float(tracked.at[det_index, "y"]),
                    "x": float(tracked.at[det_index, "x"]),
                }
                matched_detections.add(det_index)

        for det_index in det_indices:
            if det_index in matched_detections:
                continue
            track_id = next_track_id
            next_track_id += 1
            tracked.at[det_index, "track_id"] = track_id
            active_tracks[track_id] = {
                "last_frame": float(frame_id),
                "y": float(tracked.at[det_index, "y"]),
                "x": float(tracked.at[det_index, "x"]),
            }

    if tracking_params.min_track_length > 1:
        lengths = tracked.groupby("track_id").size()
        keep_ids = lengths[lengths >= tracking_params.min_track_length].index
        tracked = tracked[tracked["track_id"].isin(keep_ids)].reset_index(drop=True)

    tracked["track_id"] = tracked["track_id"].astype(int)
    return tracked


def classify_intensity(
    tracked: pd.DataFrame,
    signal_column: str = "net_intensity_p90",
    low_threshold: float | None = None,
    high_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    classified = tracked.copy()
    if classified.empty:
        classified["intensity_class"] = pd.Series(dtype=str)
        return classified, {"low_threshold": 0.0, "high_threshold": 0.0}

    values = classified[signal_column].to_numpy(dtype=float)
    if low_threshold is None:
        low_threshold = float(np.quantile(values, 0.33))
    if high_threshold is None:
        high_threshold = float(np.quantile(values, 0.66))
    if high_threshold < low_threshold:
        high_threshold = low_threshold

    labels = np.full(len(classified), "medium", dtype=object)
    labels[values <= low_threshold] = "low"
    labels[values >= high_threshold] = "high"
    classified["intensity_class"] = labels

    robust_center = float(np.median(values))
    robust_scale = float(np.median(np.abs(values - robust_center))) or 1.0
    classified["intensity_score"] = (values - robust_center) / robust_scale

    thresholds = {
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
    }
    return classified, thresholds


def make_frame_summary(tracked: pd.DataFrame) -> pd.DataFrame:
    if tracked.empty:
        return pd.DataFrame(
            columns=[
                "frame",
                "droplet_count",
                "mean_net_intensity",
                "median_net_intensity",
                "high_count",
                "medium_count",
                "low_count",
            ]
        )

    summary = (
        tracked.groupby("frame")
        .agg(
            droplet_count=("track_id", "size"),
            mean_net_intensity=("net_intensity_p90", "mean"),
            median_net_intensity=("net_intensity_p90", "median"),
            high_count=("intensity_class", lambda s: int((s == "high").sum())),
            medium_count=("intensity_class", lambda s: int((s == "medium").sum())),
            low_count=("intensity_class", lambda s: int((s == "low").sum())),
        )
        .reset_index()
        .sort_values("frame")
    )
    return summary


def make_track_summary(tracked: pd.DataFrame) -> pd.DataFrame:
    if tracked.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "start_frame",
                "end_frame",
                "frames_seen",
                "mean_net_intensity",
                "max_net_intensity",
                "dominant_class",
            ]
        )

    summary = (
        tracked.groupby("track_id")
        .agg(
            start_frame=("frame", "min"),
            end_frame=("frame", "max"),
            frames_seen=("frame", "size"),
            mean_net_intensity=("net_intensity_p90", "mean"),
            max_net_intensity=("net_intensity_p90", "max"),
            dominant_class=("intensity_class", lambda s: s.mode().iat[0]),
        )
        .reset_index()
        .sort_values("track_id")
    )
    return summary


def save_overlay_preview(
    raw_frame: np.ndarray,
    detections: pd.DataFrame,
    output_path: Path,
    truth_points: np.ndarray | None = None,
) -> None:
    image = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)

    for _, row in detections.iterrows():
        cv2.circle(
            image,
            center=(int(round(row["x"])), int(round(row["y"]))),
            radius=4,
            color=(0, 255, 0),
            thickness=1,
        )

    if truth_points is not None:
        for y, x in truth_points:
            cv2.circle(
                image,
                center=(int(round(x)), int(round(y))),
                radius=3,
                color=(0, 0, 255),
                thickness=1,
            )

    cv2.imwrite(str(output_path), image)


def get_detection_color(row: pd.Series) -> tuple[int, int, int]:
    intensity_class = row.get("intensity_class", "medium")
    if intensity_class == "high":
        return (0, 215, 255)
    if intensity_class == "low":
        return (255, 140, 0)
    return (0, 200, 0)


def draw_detection_marker(
    image: np.ndarray,
    x: float,
    y: float,
    color: tuple[int, int, int],
    radius: int = 4,
    thickness: int = 1,
) -> None:
    cv2.circle(
        image,
        center=(int(round(x)), int(round(y))),
        radius=radius,
        color=color,
        thickness=thickness,
    )


def draw_text_block(
    image: np.ndarray,
    lines: list[str],
    x: int = 8,
    y: int = 18,
    line_height: int = 18,
) -> None:
    for idx, line in enumerate(lines):
        origin = (x, y + idx * line_height)
        cv2.putText(
            image,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def build_qc_error_table(
    detections: pd.DataFrame,
    annotations: dict[int, np.ndarray],
    tolerance: float,
) -> pd.DataFrame:
    rows = []
    for frame_id in sorted(annotations):
        frame_detections = detections[detections["frame"] == frame_id].reset_index(drop=True)
        predicted = frame_detections[["y", "x"]].to_numpy(dtype=float)
        truth = annotations[frame_id]
        detail = match_points_detail(predicted, truth, tolerance=tolerance)
        rows.append(
            {
                "frame": int(frame_id),
                "predicted_count": int(len(predicted)),
                "truth_count": int(len(truth)),
                "tp": int(len(detail["tp_pred_idx"])),
                "fp": int(len(detail["fp_pred_idx"])),
                "fn": int(len(detail["fn_truth_idx"])),
                "error_score": int(len(detail["fp_pred_idx"]) + len(detail["fn_truth_idx"])),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["error_score", "fn", "fp", "frame"],
        ascending=[False, False, False, True],
    )


def render_qc_overlay_video(
    raw_frames: np.ndarray,
    detections: pd.DataFrame,
    frame_summary: pd.DataFrame,
    output_path: Path,
    fps: float,
    annotations: dict[int, np.ndarray] | None = None,
    tolerance: float = 6.0,
) -> None:
    height, width = raw_frames.shape[1:]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps or 10.0, (width, height), True)

    frame_summary_map = {
        int(row["frame"]): row for _, row in frame_summary.iterrows()
    }

    detections_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in detections.groupby("frame")
    }

    for frame_id, raw_frame in enumerate(raw_frames):
        image = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
        frame_detections = detections_by_frame.get(frame_id, pd.DataFrame())
        frame_truth = None
        detail = None
        tp_idx: set[int] = set()
        fp_idx: set[int] = set()

        if annotations is not None and frame_id in annotations:
            frame_truth = annotations[frame_id]
            predicted = (
                frame_detections[["y", "x"]].to_numpy(dtype=float)
                if not frame_detections.empty
                else np.empty((0, 2), dtype=float)
            )
            detail = match_points_detail(predicted, frame_truth, tolerance=tolerance)
            tp_idx = set(detail["tp_pred_idx"].tolist())
            fp_idx = set(detail["fp_pred_idx"].tolist())

        if not frame_detections.empty:
            for det_idx, row in frame_detections.iterrows():
                x = float(row["x"])
                y = float(row["y"])
                if det_idx in fp_idx:
                    draw_detection_marker(image, x, y, (0, 255, 255), radius=5, thickness=1)
                elif det_idx in tp_idx:
                    draw_detection_marker(image, x, y, (0, 255, 0), radius=4, thickness=1)
                else:
                    draw_detection_marker(image, x, y, get_detection_color(row), radius=4, thickness=1)

        if frame_truth is not None:
            matched_truth_idx = set(detail["matched_truth_idx"].tolist()) if detail is not None else set()
            for truth_idx, (y, x) in enumerate(frame_truth):
                if truth_idx in matched_truth_idx:
                    draw_detection_marker(image, x, y, (255, 0, 255), radius=2, thickness=1)
                else:
                    draw_detection_marker(image, x, y, (255, 0, 255), radius=6, thickness=1)
                    cv2.line(
                        image,
                        (int(round(x)) - 4, int(round(y))),
                        (int(round(x)) + 4, int(round(y))),
                        (255, 0, 255),
                        1,
                    )
                    cv2.line(
                        image,
                        (int(round(x)), int(round(y)) - 4),
                        (int(round(x)), int(round(y)) + 4),
                        (255, 0, 255),
                        1,
                    )

        summary_row = frame_summary_map.get(frame_id)
        lines = [f"frame {frame_id + 1}/{len(raw_frames)}"]
        if summary_row is not None:
            lines.append(
                "count {count} | high {high} | med {med} | low {low}".format(
                    count=int(summary_row["droplet_count"]),
                    high=int(summary_row["high_count"]),
                    med=int(summary_row["medium_count"]),
                    low=int(summary_row["low_count"]),
                )
            )
            lines.append(f"median net intensity {summary_row['median_net_intensity']:.1f}")
        if detail is not None:
            lines.append(
                "GT {gt} | TP {tp} | FP {fp} | FN {fn}".format(
                    gt=len(frame_truth),
                    tp=len(tp_idx),
                    fp=len(fp_idx),
                    fn=len(detail["fn_truth_idx"]),
                )
            )
        lines.append("green=TP, yellow=FP, magenta=GT/FN")
        draw_text_block(image, lines)
        writer.write(image)

    writer.release()


def save_qc_problem_frames(
    raw_frames: np.ndarray,
    detections: pd.DataFrame,
    annotations: dict[int, np.ndarray],
    table_output_path: Path,
    image_output_dir: Path,
    tolerance: float,
    top_k: int,
) -> pd.DataFrame:
    error_table = build_qc_error_table(detections, annotations, tolerance=tolerance)
    error_table.to_csv(table_output_path, index=False)

    detections_by_frame = {
        int(frame_id): group.reset_index(drop=True)
        for frame_id, group in detections.groupby("frame")
    }

    top_frames = error_table.head(top_k)
    for rank, (_, row) in enumerate(top_frames.iterrows(), start=1):
        frame_id = int(row["frame"])
        image = cv2.cvtColor(raw_frames[frame_id], cv2.COLOR_GRAY2BGR)
        frame_detections = detections_by_frame.get(frame_id, pd.DataFrame())
        predicted = (
            frame_detections[["y", "x"]].to_numpy(dtype=float)
            if not frame_detections.empty
            else np.empty((0, 2), dtype=float)
        )
        truth = annotations[frame_id]
        detail = match_points_detail(predicted, truth, tolerance=tolerance)

        tp_idx = set(detail["tp_pred_idx"].tolist())
        fp_idx = set(detail["fp_pred_idx"].tolist())
        matched_truth_idx = set(detail["matched_truth_idx"].tolist())
        fn_truth_idx = set(detail["fn_truth_idx"].tolist())

        if not frame_detections.empty:
            for det_idx, det_row in frame_detections.iterrows():
                x = float(det_row["x"])
                y = float(det_row["y"])
                if det_idx in tp_idx:
                    draw_detection_marker(image, x, y, (0, 255, 0), radius=4, thickness=1)
                elif det_idx in fp_idx:
                    draw_detection_marker(image, x, y, (0, 255, 255), radius=6, thickness=1)

        for truth_idx, (y, x) in enumerate(truth):
            if truth_idx in matched_truth_idx:
                draw_detection_marker(image, x, y, (255, 0, 255), radius=2, thickness=1)
            elif truth_idx in fn_truth_idx:
                draw_detection_marker(image, x, y, (255, 0, 255), radius=7, thickness=1)
                cv2.line(
                    image,
                    (int(round(x)) - 5, int(round(y)) - 5),
                    (int(round(x)) + 5, int(round(y)) + 5),
                    (255, 0, 255),
                    1,
                )
                cv2.line(
                    image,
                    (int(round(x)) - 5, int(round(y)) + 5),
                    (int(round(x)) + 5, int(round(y)) - 5),
                    (255, 0, 255),
                    1,
                )

        draw_text_block(
            image,
            [
                f"rank {rank} | frame {frame_id}",
                f"GT {len(truth)} | pred {len(predicted)}",
                f"TP {len(tp_idx)} | FP {len(fp_idx)} | FN {len(fn_truth_idx)}",
                "green=TP, yellow=FP, magenta=GT/FN",
            ],
        )
        out_name = f"qc_frame_rank_{rank:02d}_frame_{frame_id}.png"
        cv2.imwrite(str(image_output_dir / out_name), image)

    return error_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Droplet counting, tracking and glow estimation pipeline."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--tolerance", type=float, default=6.0)

    parser.add_argument("--small-sigmas", default="1.0,1.2")
    parser.add_argument("--big-sigmas", default="7,9,11")
    parser.add_argument("--thresholds", default="18,19.5,20,20.5,21,21.5,22")
    parser.add_argument("--min-distances", default="4,5,6,7")
    parser.add_argument("--net-p90-thresholds", default="24,26,28,29,30,31,32,33")
    parser.add_argument("--net-mean-thresholds", default="8,9,10,11,12,13")
    parser.add_argument(
        "--selection-metric",
        choices=["balanced", "f1", "count_mae"],
        default="balanced",
    )

    parser.add_argument("--signal-radius", type=int, default=4)
    parser.add_argument("--background-radius", type=int, default=9)

    parser.add_argument("--track-max-distance", type=float, default=10.0)
    parser.add_argument("--track-max-gap", type=int, default=3)
    parser.add_argument("--track-min-length", type=int, default=1)
    parser.add_argument("--disable-recovery-pass", action="store_true")
    parser.add_argument("--recovery-threshold-margin", type=float, default=1.0)
    parser.add_argument("--recovery-p90-margin", type=float, default=5.0)
    parser.add_argument("--recovery-mean-margin", type=float, default=3.0)
    parser.add_argument("--recovery-same-frame-radius", type=float, default=2.0)
    parser.add_argument("--recovery-temporal-radius", type=float, default=10.0)

    parser.add_argument("--intensity-low-threshold", type=float, default=None)
    parser.add_argument("--intensity-high-threshold", type=float, default=None)
    parser.add_argument("--qc-top-k", type=int, default=5)
    parser.add_argument("--skip-tuning", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    video_path = resolve_project_path(args.video)
    annotations_path = resolve_project_path(args.annotations)
    output_dir = resolve_project_path(args.output_dir)
    tables_dir = output_dir / "tables"
    qc_dir = output_dir / "qc"
    reports_dir = output_dir / "reports"

    measurement_params = MeasurementParams(
        signal_radius=args.signal_radius,
        background_radius=args.background_radius,
    )
    tracking_params = TrackingParams(
        max_distance=args.track_max_distance,
        max_gap=args.track_max_gap,
        min_track_length=args.track_min_length,
    )
    recovery_params = RecoveryParams(
        enabled=not args.disable_recovery_pass,
        threshold_margin=args.recovery_threshold_margin,
        p90_margin=args.recovery_p90_margin,
        mean_margin=args.recovery_mean_margin,
        same_frame_radius=args.recovery_same_frame_radius,
        temporal_radius=args.recovery_temporal_radius,
    )

    raw_frames, fps = load_video_grayscale(video_path)
    tables_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    annotations: dict[int, np.ndarray] = {}
    if annotations_path.exists():
        annotations = load_annotations(annotations_path)

    if annotations and not args.skip_tuning:
        _, tuning_results = tune_detector(
            raw_frames=raw_frames,
            annotations=annotations,
            measurement_params=measurement_params,
            small_sigmas=parse_grid(args.small_sigmas, float),
            big_sigmas=parse_grid(args.big_sigmas, float),
            thresholds=parse_grid(args.thresholds, float),
            min_distances=parse_grid(args.min_distances, int),
            net_p90_thresholds=parse_grid(args.net_p90_thresholds, float),
            net_mean_thresholds=parse_grid(args.net_mean_thresholds, float),
            tolerance=args.tolerance,
        )
        best_params, tuning_results = select_best_params(
            results=tuning_results,
            selection_metric=args.selection_metric,
        )
        tuning_results.to_csv(tables_dir / "tuning_results.csv", index=False)
    else:
        best_params = DetectorParams()
        tuning_results = None

    detections, _ = detect_video(
        raw_frames=raw_frames,
        detector_params=best_params,
        measurement_params=measurement_params,
    )
    detections = recover_temporal_candidates(
        raw_frames=raw_frames,
        strong_detections=detections,
        detector_params=best_params,
        measurement_params=measurement_params,
        recovery_params=recovery_params,
    )

    evaluation_metrics = None
    per_frame_eval = None
    if annotations:
        evaluation_metrics, per_frame_eval = evaluate_detections(
            detections=detections[["frame", "y", "x"]],
            annotations=annotations,
            tolerance=args.tolerance,
        )
        per_frame_eval.to_csv(tables_dir / "evaluation_per_frame.csv", index=False)

    tracked = track_detections(detections=detections, tracking_params=tracking_params)
    classified, class_thresholds = classify_intensity(
        tracked=tracked,
        signal_column="net_intensity_p90",
        low_threshold=args.intensity_low_threshold,
        high_threshold=args.intensity_high_threshold,
    )

    frame_summary = make_frame_summary(classified)
    track_summary = make_track_summary(classified)

    classified.to_csv(tables_dir / "detections_tracked.csv", index=False)
    frame_summary.to_csv(tables_dir / "frame_summary.csv", index=False)
    track_summary.to_csv(tables_dir / "track_summary.csv", index=False)

    render_qc_overlay_video(
        raw_frames=raw_frames,
        detections=classified,
        frame_summary=frame_summary,
        output_path=qc_dir / "qc_overlay.mp4",
        fps=fps,
        annotations=annotations if annotations else None,
        tolerance=args.tolerance,
    )

    if annotations:
        preview_frame = min(annotations)
        preview_detections = classified[classified["frame"] == preview_frame]
        save_overlay_preview(
            raw_frame=raw_frames[preview_frame],
            detections=preview_detections,
            truth_points=annotations[preview_frame],
            output_path=qc_dir / f"preview_frame_{preview_frame}.png",
        )
        qc_error_table = save_qc_problem_frames(
            raw_frames=raw_frames,
            detections=classified,
            annotations=annotations,
            table_output_path=tables_dir / "qc_problem_frames.csv",
            image_output_dir=qc_dir,
            tolerance=args.tolerance,
            top_k=args.qc_top_k,
        )
    else:
        qc_error_table = None

    summary = {
        "project_dir": str(PROJECT_DIR),
        "video": str(video_path),
        "fps": fps,
        "frame_count": int(raw_frames.shape[0]),
        "detector_params": asdict(best_params),
        "selection_metric": args.selection_metric,
        "measurement_params": asdict(measurement_params),
        "tracking_params": asdict(tracking_params),
        "recovery_params": asdict(recovery_params),
        "intensity_thresholds": class_thresholds,
        "detections_total": int(len(classified)),
        "tracks_total": int(classified["track_id"].nunique()) if not classified.empty else 0,
        "evaluation": evaluation_metrics,
        "qc": {
            "overlay_video": str(qc_dir / "qc_overlay.mp4"),
            "problem_frames_csv": str(tables_dir / "qc_problem_frames.csv")
            if annotations
            else None,
            "top_problem_frames": qc_error_table.head(args.qc_top_k).to_dict(orient="records")
            if qc_error_table is not None
            else [],
        },
    }

    with open(reports_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("Run summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if tuning_results is not None:
        print("\nTop tuning results")
        print(tuning_results.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
