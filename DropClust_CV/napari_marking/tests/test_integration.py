from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"


def write_synthetic_video(video_path: Path, num_frames: int = 4) -> None:
    frame_size = (64, 64)
    codecs = ["MJPG", "XVID"]
    writer = None
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(str(video_path), fourcc, 5.0, frame_size, True)
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        raise RuntimeError("Could not create synthetic video writer.")

    for frame_id in range(num_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + frame_id, 20), 3, (255, 255, 255), -1)
        cv2.circle(frame, (45, 42 - frame_id), 3, (220, 220, 220), -1)
        writer.write(frame)
    writer.release()


def write_annotations(csv_path: Path, num_frames: int = 4) -> None:
    rows = []
    for frame_id in range(num_frames):
        rows.append({"frame": frame_id, "y": 20.0, "x": float(20 + frame_id)})
        rows.append({"frame": frame_id, "y": float(42 - frame_id), "x": 45.0})
    pd.DataFrame(rows).to_csv(csv_path, index=False)


class PipelineIntegrationTests(unittest.TestCase):
    def test_droplet_cv_cli_generates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            video_path = tmp / "synthetic.avi"
            annotations_path = tmp / "synthetic.csv"
            output_dir = tmp / "output"
            write_synthetic_video(video_path)
            write_annotations(annotations_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SRC_DIR / "droplet_cv.py"),
                    "--video",
                    str(video_path),
                    "--annotations",
                    str(annotations_path),
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    "baseline",
                    "--skip-tuning",
                    "--qc-top-k",
                    "2",
                ],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)

            summary_path = output_dir / "reports" / "run_summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "baseline")
            self.assertEqual(summary["frame_count"], 4)
            self.assertIn("evaluation", summary)
            self.assertTrue((output_dir / "tables" / "detections_tracked.csv").exists())
            self.assertTrue((output_dir / "tables" / "frame_summary.csv").exists())
            self.assertTrue((output_dir / "qc" / "qc_overlay.mp4").exists())

    def test_mlops_batch_cli_generates_manifest_and_job_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_videos = tmp / "input" / "videos"
            input_annotations = tmp / "input" / "annotations"
            output_dir = tmp / "output"
            input_videos.mkdir(parents=True)
            input_annotations.mkdir(parents=True)

            video_path = input_videos / "sample.avi"
            write_synthetic_video(video_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SRC_DIR / "mlops_batch.py"),
                    "--input-videos",
                    str(input_videos),
                    "--input-annotations",
                    str(input_annotations),
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    "balanced",
                    "--overwrite",
                ],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)

            manifest_path = output_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["status"], "ok")
            self.assertEqual(manifest[0]["mode"], "balanced")
            self.assertTrue((output_dir / "sample" / "reports" / "run_summary.json").exists())
            self.assertTrue((output_dir / "sample.stdout.log").exists())
            self.assertTrue((output_dir / "sample.stderr.log").exists())


if __name__ == "__main__":
    unittest.main()
