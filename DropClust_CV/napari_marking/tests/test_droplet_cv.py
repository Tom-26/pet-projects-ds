from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import droplet_cv  # noqa: E402


def sample_detection_rows(rows: list[dict[str, float | int]]) -> pd.DataFrame:
    base = pd.DataFrame(rows)
    required_defaults = {
        "frame": 0,
        "y": 0.0,
        "x": 0.0,
        "filtered_peak": 0.0,
        "net_intensity_mean": 0.0,
        "net_intensity_peak": 0.0,
        "net_intensity_p90": 0.0,
        "raw_signal_mean": 0.0,
        "raw_signal_peak": 0.0,
        "raw_signal_p90": 0.0,
        "raw_background_median": 0.0,
    }
    for column, default in required_defaults.items():
        if column not in base.columns:
            base[column] = default
    return base


class DropletCvUtilityTests(unittest.TestCase):
    def test_build_annotation_blocks_groups_contiguous_frames(self) -> None:
        annotations = {
            10: np.array([[1.0, 1.0]]),
            0: np.array([[0.0, 0.0]]),
            1: np.array([[0.0, 1.0]]),
            3: np.array([[1.0, 0.0]]),
            7: np.array([[2.0, 2.0]]),
            8: np.array([[2.0, 3.0]]),
        }
        blocks = droplet_cv.build_annotation_blocks(annotations)
        self.assertEqual(blocks, [[0, 1], [3], [7, 8], [10]])

    def test_make_temporal_holdout_split_uses_alternating_blocks(self) -> None:
        annotations = {
            0: np.array([[0.0, 0.0]]),
            1: np.array([[1.0, 1.0]]),
            4: np.array([[2.0, 2.0]]),
            5: np.array([[3.0, 3.0]]),
            8: np.array([[4.0, 4.0]]),
        }
        split = droplet_cv.make_temporal_holdout_split(annotations, holdout_block_offset=1)
        self.assertEqual(split["blocks"], [[0, 1], [4, 5], [8]])
        self.assertEqual(split["holdout_blocks"], [[4, 5]])
        self.assertEqual(split["train_blocks"], [[0, 1], [8]])
        self.assertEqual(split["holdout_frames"], [4, 5])
        self.assertEqual(split["train_frames"], [0, 1, 8])

    def test_load_annotations_groups_points_by_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "annotations.csv"
            pd.DataFrame(
                [
                    {"frame": 0, "y": 10.0, "x": 20.0},
                    {"frame": 0, "y": 11.0, "x": 21.0},
                    {"frame": 2, "y": 30.0, "x": 40.0},
                ]
            ).to_csv(csv_path, index=False)
            annotations = droplet_cv.load_annotations(csv_path)

        self.assertEqual(sorted(annotations), [0, 2])
        np.testing.assert_array_equal(
            annotations[0],
            np.array([[10.0, 20.0], [11.0, 21.0]], dtype=float),
        )
        np.testing.assert_array_equal(
            annotations[2],
            np.array([[30.0, 40.0]], dtype=float),
        )

    def test_temporal_median_filter_video_handles_edges(self) -> None:
        frames = np.array([[[0.0]], [[10.0]], [[100.0]]], dtype=np.float32)
        filtered = droplet_cv.temporal_median_filter_video(frames, window_size=3)
        np.testing.assert_array_equal(
            filtered,
            np.array([[[0.0]], [[10.0]], [[100.0]]], dtype=np.float32),
        )

    def test_make_response_video_rejects_unknown_branch(self) -> None:
        frames = np.zeros((2, 4, 4), dtype=np.uint8)
        params = droplet_cv.preset_baseline_params()
        with self.assertRaises(ValueError):
            droplet_cv.make_response_video(frames, params, branch_name="unknown")

    def test_point_array_and_min_dist_handle_empty_inputs(self) -> None:
        empty_points = droplet_cv.point_array(pd.DataFrame(columns=["y", "x"]))
        self.assertEqual(empty_points.shape, (0, 2))
        self.assertEqual(droplet_cv.min_dist(np.array([1.0, 1.0]), empty_points), float("inf"))

        df = pd.DataFrame([{"y": 1.0, "x": 2.0}, {"y": 4.0, "x": 6.0}])
        points = droplet_cv.point_array(df)
        self.assertEqual(points.shape, (2, 2))
        self.assertAlmostEqual(droplet_cv.min_dist(np.array([1.0, 3.0]), points), 1.0)

    def test_ensemble_high_recall_union_deduplicates_nearby_bgsub_points(self) -> None:
        baseline = sample_detection_rows(
            [
                {"frame": 0, "y": 10.0, "x": 10.0},
                {"frame": 1, "y": 5.0, "x": 5.0},
            ]
        )
        bgsub = sample_detection_rows(
            [
                {"frame": 0, "y": 10.5, "x": 10.5},
                {"frame": 0, "y": 20.0, "x": 20.0},
            ]
        )
        fused = droplet_cv.ensemble_fuse_high_recall_union(
            baseline,
            bgsub,
            droplet_cv.HighRecallUnionParams(same_frame_match_radius=1.0),
        )

        self.assertEqual(len(fused), 3)
        self.assertCountEqual(fused["source_branch"].tolist(), ["baseline", "baseline", "bgsub_union"])
        self.assertTrue(((fused["frame"] == 0) & (fused["y"] == 20.0) & (fused["x"] == 20.0)).any())

    def test_merge_branch_candidates_combines_overlapping_detections(self) -> None:
        baseline = sample_detection_rows(
            [
                {
                    "frame": 0,
                    "y": 10.0,
                    "x": 10.0,
                    "filtered_peak": 5.0,
                    "net_intensity_mean": 20.0,
                    "net_intensity_peak": 30.0,
                    "net_intensity_p90": 25.0,
                    "raw_signal_mean": 100.0,
                    "raw_signal_peak": 120.0,
                    "raw_signal_p90": 110.0,
                    "raw_background_median": 10.0,
                }
            ]
        )
        bgsub = sample_detection_rows(
            [
                {
                    "frame": 0,
                    "y": 10.5,
                    "x": 10.5,
                    "filtered_peak": 9.0,
                    "net_intensity_mean": 21.0,
                    "net_intensity_peak": 31.0,
                    "net_intensity_p90": 26.0,
                    "raw_signal_mean": 101.0,
                    "raw_signal_peak": 121.0,
                    "raw_signal_p90": 111.0,
                    "raw_background_median": 11.0,
                },
                {
                    "frame": 0,
                    "y": 30.0,
                    "x": 30.0,
                    "filtered_peak": 7.0,
                    "net_intensity_mean": 18.0,
                    "net_intensity_peak": 28.0,
                    "net_intensity_p90": 23.0,
                },
            ]
        )
        merged = droplet_cv.merge_branch_candidates(baseline, bgsub, merge_radius=1.0)

        self.assertEqual(len(merged), 2)
        overlapping = merged.iloc[0]
        self.assertEqual(overlapping["source_baseline"], 1)
        self.assertEqual(overlapping["source_bgsub"], 1)
        self.assertEqual(overlapping["branch_agreement"], 1)
        self.assertEqual(overlapping["filtered_peak"], 9.0)
        self.assertEqual(overlapping["y"], 10.5)
        self.assertEqual(overlapping["x"], 10.5)

        bg_only = merged.iloc[1]
        self.assertEqual(bg_only["source_baseline"], 0)
        self.assertEqual(bg_only["source_bgsub"], 1)
        self.assertEqual(bg_only["branch_agreement"], 0)

    def test_add_temporal_support_features_marks_neighbor_support(self) -> None:
        candidates = pd.DataFrame(
            [
                {"frame": 0, "y": 10.0, "x": 10.0, "source_baseline": 1, "source_bgsub": 0},
                {"frame": 1, "y": 10.4, "x": 10.4, "source_baseline": 1, "source_bgsub": 0},
                {"frame": 2, "y": 40.0, "x": 40.0, "source_baseline": 0, "source_bgsub": 1},
            ]
        )
        enriched = droplet_cv.add_temporal_support_features(candidates, support_radius=1.0)

        middle = enriched[enriched["frame"] == 1].iloc[0]
        self.assertEqual(middle["base_prev_support"], 1.0)
        self.assertEqual(middle["base_next_support"], 0.0)
        self.assertGreater(middle["temporal_support_sum"], 0.0)
        self.assertEqual(middle["temporal_support_any"], 1.0)

    def test_select_best_params_supports_all_selection_modes(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "small_sigma": 1.0,
                    "big_sigma": 7.0,
                    "threshold_abs": 20.0,
                    "min_distance": 4,
                    "min_net_intensity_p90": 25.0,
                    "min_net_intensity_mean": 8.0,
                    "precision": 0.90,
                    "recall": 0.80,
                    "f1": 0.85,
                    "count_mae": 3.0,
                },
                {
                    "small_sigma": 1.2,
                    "big_sigma": 11.0,
                    "threshold_abs": 21.0,
                    "min_distance": 5,
                    "min_net_intensity_p90": 31.0,
                    "min_net_intensity_mean": 9.0,
                    "precision": 0.88,
                    "recall": 0.88,
                    "f1": 0.88,
                    "count_mae": 12.0,
                },
            ]
        )

        best_f1, _ = droplet_cv.select_best_params(results, selection_metric="f1")
        self.assertEqual(best_f1.big_sigma, 11.0)

        best_mae, _ = droplet_cv.select_best_params(results, selection_metric="count_mae")
        self.assertEqual(best_mae.big_sigma, 7.0)

        best_balanced, ranked = droplet_cv.select_best_params(results, selection_metric="balanced")
        self.assertEqual(best_balanced.big_sigma, 11.0)
        self.assertIn("balanced_score", ranked.columns)

    def test_classify_intensity_respects_manual_thresholds(self) -> None:
        tracked = pd.DataFrame(
            [
                {"track_id": 1, "frame": 0, "net_intensity_p90": 10.0},
                {"track_id": 2, "frame": 0, "net_intensity_p90": 20.0},
                {"track_id": 3, "frame": 1, "net_intensity_p90": 30.0},
            ]
        )
        classified, thresholds = droplet_cv.classify_intensity(
            tracked,
            low_threshold=15.0,
            high_threshold=25.0,
        )

        self.assertEqual(thresholds["low_threshold"], 15.0)
        self.assertEqual(thresholds["high_threshold"], 25.0)
        self.assertEqual(classified["intensity_class"].tolist(), ["low", "medium", "high"])
        self.assertIn("intensity_score", classified.columns)

    def test_make_frame_and_track_summary_aggregate_counts(self) -> None:
        tracked = pd.DataFrame(
            [
                {"frame": 0, "track_id": 1, "net_intensity_p90": 10.0, "intensity_class": "low"},
                {"frame": 0, "track_id": 2, "net_intensity_p90": 30.0, "intensity_class": "high"},
                {"frame": 1, "track_id": 1, "net_intensity_p90": 20.0, "intensity_class": "medium"},
                {"frame": 1, "track_id": 2, "net_intensity_p90": 35.0, "intensity_class": "high"},
            ]
        )

        frame_summary = droplet_cv.make_frame_summary(tracked)
        track_summary = droplet_cv.make_track_summary(tracked)

        self.assertEqual(frame_summary.loc[frame_summary["frame"] == 0, "droplet_count"].iat[0], 2)
        self.assertEqual(frame_summary.loc[frame_summary["frame"] == 0, "high_count"].iat[0], 1)
        self.assertEqual(frame_summary.loc[frame_summary["frame"] == 1, "medium_count"].iat[0], 1)

        track_one = track_summary.loc[track_summary["track_id"] == 1].iloc[0]
        self.assertEqual(track_one["start_frame"], 0)
        self.assertEqual(track_one["end_frame"], 1)
        self.assertEqual(track_one["frames_seen"], 2)

    def test_parse_grid_and_resolve_project_path(self) -> None:
        self.assertEqual(droplet_cv.parse_grid("1, 2,3", int), [1, 2, 3])
        absolute = Path("/tmp/example")
        self.assertEqual(droplet_cv.resolve_project_path(absolute), absolute)
        relative = Path("data/raw/input.avi")
        self.assertEqual(droplet_cv.resolve_project_path(relative), droplet_cv.PROJECT_DIR / relative)


if __name__ == "__main__":
    unittest.main()
