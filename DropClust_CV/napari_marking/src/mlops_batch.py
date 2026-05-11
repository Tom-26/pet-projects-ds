from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal batch runner for droplet CV inference."
    )
    parser.add_argument("--input-videos", type=Path, default=Path("mlops/input/videos"))
    parser.add_argument("--input-annotations", type=Path, default=Path("mlops/input/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("mlops/output"))
    parser.add_argument(
        "--mode",
        choices=["balanced", "high-recall", "baseline"],
        default="balanced",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent.parent / path).resolve()


def discover_videos(input_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    input_videos = resolve(args.input_videos)
    input_annotations = resolve(args.input_annotations)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_videos.mkdir(parents=True, exist_ok=True)
    input_annotations.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(input_videos)
    manifest_rows = []

    if not videos:
        print(f"No videos found in: {input_videos}")
        return

    for video_path in videos:
        job_name = video_path.stem
        job_output_dir = output_dir / job_name
        if job_output_dir.exists() and not args.overwrite:
            manifest_rows.append(
                {
                    "job_name": job_name,
                    "video": str(video_path),
                    "status": "skipped_existing",
                    "output_dir": str(job_output_dir),
                    "mode": args.mode,
                }
            )
            continue

        annotation_path = input_annotations / f"{job_name}.csv"
        command = [
            sys.executable,
            str(project_dir / "src/droplet_cv.py"),
            "--video",
            str(video_path),
            "--output-dir",
            str(job_output_dir),
            "--mode",
            args.mode,
            "--skip-tuning",
        ]
        if annotation_path.exists():
            command.extend(["--annotations", str(annotation_path)])
        else:
            command.append("--no-annotations")

        completed = subprocess.run(
            command,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
        )

        run_summary_path = job_output_dir / "reports" / "run_summary.json"
        summary = {}
        if run_summary_path.exists():
            summary = json.loads(run_summary_path.read_text(encoding="utf-8"))

        manifest_rows.append(
            {
                "job_name": job_name,
                "video": str(video_path),
                "annotation": str(annotation_path) if annotation_path.exists() else "",
                "status": "ok" if completed.returncode == 0 else "failed",
                "return_code": completed.returncode,
                "mode": args.mode,
                "output_dir": str(job_output_dir),
                "detections_total": summary.get("detections_total"),
                "tracks_total": summary.get("tracks_total"),
                "precision": (summary.get("evaluation") or {}).get("precision"),
                "recall": (summary.get("evaluation") or {}).get("recall"),
                "f1": (summary.get("evaluation") or {}).get("f1"),
            }
        )

        log_prefix = output_dir / f"{job_name}"
        (log_prefix.with_suffix(".stdout.log")).write_text(completed.stdout, encoding="utf-8")
        (log_prefix.with_suffix(".stderr.log")).write_text(completed.stderr, encoding="utf-8")

    manifest = manifest_rows
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
