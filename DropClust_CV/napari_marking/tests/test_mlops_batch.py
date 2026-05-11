from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mlops_batch  # noqa: E402


class MlopsBatchTests(unittest.TestCase):
    def test_resolve_keeps_absolute_paths(self) -> None:
        absolute = Path("/tmp/mlops-output")
        self.assertEqual(mlops_batch.resolve(absolute), absolute)

    def test_resolve_expands_project_relative_paths(self) -> None:
        resolved = mlops_batch.resolve(Path("mlops/output"))
        self.assertEqual(resolved, PROJECT_DIR / "mlops" / "output")

    def test_discover_videos_filters_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.avi").write_bytes(b"")
            (root / "b.MP4").write_bytes(b"")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "c.mov").write_bytes(b"")

            videos = mlops_batch.discover_videos(root)

        self.assertEqual([path.name for path in videos], ["a.avi", "b.MP4"])

    def test_make_parser_defaults_to_balanced_mode(self) -> None:
        parser = mlops_batch.make_parser()
        args = parser.parse_args([])
        self.assertEqual(args.mode, "balanced")
        self.assertEqual(args.input_videos, Path("mlops/input/videos"))
        self.assertEqual(args.output_dir, Path("mlops/output"))


if __name__ == "__main__":
    unittest.main()
