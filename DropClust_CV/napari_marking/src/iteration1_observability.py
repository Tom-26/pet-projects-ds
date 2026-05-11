from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import droplet_cv as dcv


BRANCHES = [
    ("baseline_bandpass", "Baseline bandpass"),
    ("background_subtracted", "Local background subtraction"),
    ("temporal_median_bandpass", "Temporal median + bandpass"),
]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iteration 1 detector observability and branch comparison."
    )
    parser.add_argument("--video", type=Path, default=dcv.DEFAULT_VIDEO_PATH)
    parser.add_argument("--annotations", type=Path, default=dcv.DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/iteration1"),
    )
    parser.add_argument(
        "--selection-metric",
        choices=["balanced", "f1", "count_mae"],
        default="balanced",
    )
    parser.add_argument("--holdout-block-offset", type=int, default=1)
    return parser


def subset_annotations(
    annotations: dict[int, object],
    frame_ids: list[int],
) -> dict[int, object]:
    return {frame_id: annotations[frame_id] for frame_id in frame_ids}


def evaluate_branch_split(
    raw_frames,
    annotations_subset,
    frame_ids,
    params,
    measurement_params,
    recovery_params,
    branch_name,
):
    detections, _ = dcv.detect_video(
        raw_frames=raw_frames,
        detector_params=params,
        measurement_params=measurement_params,
        frame_indices=frame_ids,
        branch_name=branch_name,
    )
    detections = dcv.recover_temporal_candidates(
        raw_frames=raw_frames,
        strong_detections=detections,
        detector_params=params,
        measurement_params=measurement_params,
        recovery_params=recovery_params,
        branch_name=branch_name,
    )
    detections = detections[detections["frame"].isin(frame_ids)].reset_index(drop=True)
    metrics, per_frame = dcv.evaluate_detections(
        detections=detections[["frame", "y", "x"]],
        annotations=annotations_subset,
        tolerance=6.0,
    )
    return metrics, per_frame


def branch_color(branch_name: str) -> str:
    palette = {
        "baseline_bandpass": "#1f77b4",
        "background_subtracted": "#ff7f0e",
        "temporal_median_bandpass": "#2ca02c",
    }
    return palette.get(branch_name, "#444444")


def branch_short_label(branch_name: str) -> str:
    short_labels = {
        "baseline_bandpass": "baseline",
        "background_subtracted": "bg-sub",
        "temporal_median_bandpass": "temp-med",
    }
    return short_labels.get(branch_name, branch_name)


def render_holdout_svg(metrics_df: pd.DataFrame, output_path: Path) -> None:
    holdout = metrics_df[metrics_df["split"] == "holdout"].copy()
    holdout = holdout.sort_values("branch_name").reset_index(drop=True)

    width = 1080
    height = 560
    left = 90
    top = 70
    chart_width = 380
    chart_height = 340
    gap = 80
    bottom = top + chart_height

    metric_names = ["precision", "recall", "f1"]
    max_mae = max(holdout["count_mae"].max() * 1.1, 1.0)

    def y_scale_score(value: float) -> float:
        return bottom - chart_height * value

    def y_scale_mae(value: float) -> float:
        return bottom - chart_height * (value / max_mae)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="40" y="36" font-family="Arial, sans-serif" font-size="24" font-weight="700">Iteration 1 Detector Comparison (Holdout)</text>',
        '<text x="40" y="58" font-family="Arial, sans-serif" font-size="13" fill="#555">Left: precision / recall / F1. Right: count MAE (lower is better).</text>',
    ]

    # Left chart
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{left + chart_width}" y2="{bottom}" stroke="#333" stroke-width="1.2"/>',
        ]
    )
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_scale_score(tick)
        lines.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+chart_width}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(f'<text x="{left-14}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#555">{tick:.2f}</text>')

    group_width = chart_width / max(len(holdout), 1)
    bar_width = group_width / 4.5
    for idx, row in holdout.iterrows():
        group_x = left + idx * group_width + group_width * 0.15
        for metric_idx, metric_name in enumerate(metric_names):
            value = float(row[metric_name])
            x = group_x + metric_idx * bar_width * 1.2
            y = y_scale_score(value)
            h = bottom - y
            fill = branch_color(row["branch_name"])
            opacity = 0.45 + 0.18 * metric_idx
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{fill}" fill-opacity="{opacity:.2f}" rx="2"/>'
            )
        lines.append(
            f'<text x="{group_x + bar_width*1.4:.1f}" y="{bottom+18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">{row["branch_short"]}</text>'
        )

    lines.append(
        f'<text x="{left + chart_width/2:.1f}" y="{bottom+42}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Branch</text>'
    )
    lines.append(
        f'<text x="{left + chart_width/2:.1f}" y="{top-16}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="700">Precision / Recall / F1</text>'
    )

    # Right chart for count_mae
    right_left = left + chart_width + gap
    right_width = 380
    lines.extend(
        [
            f'<line x1="{right_left}" y1="{top}" x2="{right_left}" y2="{bottom}" stroke="#333" stroke-width="1.2"/>',
            f'<line x1="{right_left}" y1="{bottom}" x2="{right_left + right_width}" y2="{bottom}" stroke="#333" stroke-width="1.2"/>',
        ]
    )
    for tick in range(0, int(max_mae) + 2, 2):
        y = y_scale_mae(float(tick))
        lines.append(f'<line x1="{right_left-5}" y1="{y:.1f}" x2="{right_left+right_width}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(f'<text x="{right_left-14}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#555">{tick}</text>')

    mae_group_width = right_width / max(len(holdout), 1)
    mae_bar_width = mae_group_width * 0.55
    for idx, row in holdout.iterrows():
        value = float(row["count_mae"])
        x = right_left + idx * mae_group_width + mae_group_width * 0.22
        y = y_scale_mae(value)
        h = bottom - y
        fill = branch_color(row["branch_name"])
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{mae_bar_width:.1f}" height="{h:.1f}" fill="{fill}" rx="2"/>'
        )
        lines.append(
            f'<text x="{x + mae_bar_width/2:.1f}" y="{bottom+18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">{row["branch_short"]}</text>'
        )
        lines.append(
            f'<text x="{x + mae_bar_width/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333">{value:.2f}</text>'
        )

    lines.append(
        f'<text x="{right_left + right_width/2:.1f}" y="{bottom+42}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Branch</text>'
    )
    lines.append(
        f'<text x="{right_left + right_width/2:.1f}" y="{top-16}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="700">Count MAE</text>'
    )

    # Legend
    legend_x = 60
    legend_y = height - 70
    for idx, metric_name in enumerate(metric_names):
        opacity = 0.45 + 0.18 * idx
        x = legend_x + idx * 120
        lines.append(f'<rect x="{x}" y="{legend_y}" width="16" height="16" fill="#1f77b4" fill-opacity="{opacity:.2f}" rx="2"/>')
        lines.append(f'<text x="{x+24}" y="{legend_y+13}" font-family="Arial, sans-serif" font-size="12">{metric_name}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(metrics_df: pd.DataFrame, params_df: pd.DataFrame, split_info: dict, output_path: Path) -> None:
    holdout = metrics_df[metrics_df["split"] == "holdout"].copy()
    holdout = holdout.sort_values("f1", ascending=False).reset_index(drop=True)

    lines = [
        "# Iteration 1 Detector Observability",
        "",
        "## Temporal split",
        "",
        f"- blocks: `{split_info['blocks']}`",
        f"- train frames: `{split_info['train_frames']}`",
        f"- holdout frames: `{split_info['holdout_frames']}`",
        "",
        "## Holdout summary",
        "",
        "| Branch | Precision | Recall | F1 | Count MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in holdout.iterrows():
        lines.append(
            f"| {row['branch_label']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['count_mae']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Best params",
            "",
            "| Branch | small_sigma | big_sigma | threshold_abs | min_distance | min_net_intensity_p90 | min_net_intensity_mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in params_df.iterrows():
        lines.append(
            f"| {row['branch_label']} | {row['small_sigma']:.1f} | {row['big_sigma']:.1f} | {row['threshold_abs']:.1f} | {int(row['min_distance'])} | {row['min_net_intensity_p90']:.1f} | {row['min_net_intensity_mean']:.1f} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    video_path = dcv.resolve_project_path(args.video)
    annotations_path = dcv.resolve_project_path(args.annotations)
    output_dir = dcv.resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_frames, _ = dcv.load_video_grayscale(video_path)
    annotations = dcv.load_annotations(annotations_path)
    measurement_params = dcv.MeasurementParams()
    recovery_params = dcv.RecoveryParams()

    split_info = dcv.make_temporal_holdout_split(
        annotations=annotations,
        holdout_block_offset=args.holdout_block_offset,
    )
    train_annotations = subset_annotations(annotations, split_info["train_frames"])
    holdout_annotations = subset_annotations(annotations, split_info["holdout_frames"])

    metrics_rows = []
    params_rows = []
    holdout_per_frame_rows = []

    for branch_name, branch_label in BRANCHES:
        _, tuning_results = dcv.tune_detector(
            raw_frames=raw_frames,
            annotations=train_annotations,
            measurement_params=measurement_params,
            small_sigmas=[1.0, 1.2],
            big_sigmas=[9.0, 11.0],
            thresholds=[19.5, 20.0, 20.5, 21.0],
            min_distances=[5, 6],
            net_p90_thresholds=[27.0, 29.0, 31.0],
            net_mean_thresholds=[9.0, 11.0, 13.0],
            tolerance=6.0,
            branch_name=branch_name,
        )
        params, ranked = dcv.select_best_params(tuning_results, args.selection_metric)

        params_rows.append(
            {
                "branch_name": branch_name,
                "branch_label": branch_label,
                **asdict(params),
            }
        )
        ranked.head(25).to_csv(output_dir / f"{branch_name}_top_tuning_results.csv", index=False)

        split_map = {
            "train": (split_info["train_frames"], train_annotations),
            "holdout": (split_info["holdout_frames"], holdout_annotations),
            "all_annotated": (sorted(annotations), annotations),
        }
        for split_name, (frame_ids, annotations_subset) in split_map.items():
            metrics, per_frame = evaluate_branch_split(
                raw_frames=raw_frames,
                annotations_subset=annotations_subset,
                frame_ids=frame_ids,
                params=params,
                measurement_params=measurement_params,
                recovery_params=recovery_params,
                branch_name=branch_name,
            )
            metrics_rows.append(
                {
                    "branch_name": branch_name,
                    "branch_label": branch_label,
                    "branch_short": branch_short_label(branch_name),
                    "split": split_name,
                    **metrics,
                }
            )
            if split_name == "holdout":
                branch_holdout = per_frame.copy()
                branch_holdout["branch_name"] = branch_name
                branch_holdout["branch_label"] = branch_label
                branch_holdout["error_score"] = branch_holdout["fp"] + branch_holdout["fn"]
                holdout_per_frame_rows.append(branch_holdout)

    metrics_df = pd.DataFrame(metrics_rows)
    params_df = pd.DataFrame(params_rows)
    holdout_per_frame_df = pd.concat(holdout_per_frame_rows, ignore_index=True)

    metrics_df.to_csv(output_dir / "branch_metrics.csv", index=False)
    params_df.to_csv(output_dir / "branch_best_params.csv", index=False)
    holdout_per_frame_df.to_csv(output_dir / "holdout_per_frame.csv", index=False)

    split_summary = {
        "blocks": split_info["blocks"],
        "train_frames": split_info["train_frames"],
        "holdout_frames": split_info["holdout_frames"],
        "selection_metric": args.selection_metric,
    }
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as fh:
        json.dump(split_summary, fh, ensure_ascii=False, indent=2)

    render_holdout_svg(metrics_df, output_dir / "holdout_metrics.svg")
    write_markdown_report(metrics_df, params_df, split_summary, output_dir / "report.md")

    best_holdout = (
        metrics_df[metrics_df["split"] == "holdout"]
        .sort_values(["f1", "precision", "recall"], ascending=[False, False, False])
        .iloc[0]
    )
    summary = {
        "best_holdout_branch": best_holdout["branch_name"],
        "best_holdout_metrics": {
            "precision": float(best_holdout["precision"]),
            "recall": float(best_holdout["recall"]),
            "f1": float(best_holdout["f1"]),
            "count_mae": float(best_holdout["count_mae"]),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("Iteration 1 observability saved to:", output_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
