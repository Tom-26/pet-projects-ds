# Minimal MLOps Flow

Drop videos into `mlops/input/videos/`.

Optional:
- add matching annotation CSV into `mlops/input/annotations/`
- filename must match the video stem, for example:
  - `videos/sample_01.avi`
  - `annotations/sample_01.csv`
- if no CSV exists, runner executes pure inference and does not compute metrics

Run locally:

```bash
./venv/bin/python src/mlops_batch.py --mode balanced
```

Or through the project `Makefile`:

```bash
make batch
make batch-highrecall
```

Run in Docker:

```bash
docker compose run --rm droplet-cv --mode balanced
```

Outputs appear in:

- `mlops/output/<video_stem>/tables`
- `mlops/output/<video_stem>/qc`
- `mlops/output/<video_stem>/reports`

Batch logs and manifest:

- `mlops/output/manifest.json`
- `mlops/output/<video_stem>.stdout.log`
- `mlops/output/<video_stem>.stderr.log`

Modes:

- `balanced` - best current precision/recall trade-off
- `high-recall` - more aggressive detection, better if missing droplets is worse than extra FP
- `baseline` - plain baseline detector without ensemble/rescoring
