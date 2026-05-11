PIPELINE_PYTHON ?= ./venv/bin/python
GUI_PYTHON ?= python
NAPARI_PYTHON ?= $(GUI_PYTHON)
VIDEO ?= data/raw/input.avi
ANNOTATIONS ?= data/annotations/manual_points.csv
OUTPUT_DIR ?= artifacts/make_infer
BATCH_OUTPUT_DIR ?= mlops/output
DOCKER_SERVICE ?= droplet-cv

.PHONY: help infer infer-highrecall infer-baseline eval eval-highrecall batch batch-highrecall label gui test docker-build docker-batch docker-batch-highrecall

help:
	@printf "%s\n" \
	"make infer                 # quick balanced inference without annotations" \
	"make infer-highrecall      # quick high-recall inference without annotations" \
	"make infer-baseline        # quick baseline inference without annotations" \
	"make eval                  # balanced run with local annotations" \
	"make eval-highrecall       # high-recall run with local annotations" \
	"make batch                 # run batch pipeline over mlops/input/videos" \
	"make batch-highrecall      # run batch pipeline in high-recall mode" \
	"make label                 # open napari labeling UI" \
	"make gui                   # open desktop GUI" \
	"make test                  # run unit tests" \
	"make docker-build          # build docker image" \
	"make docker-batch          # run batch pipeline in docker (balanced)" \
	"make docker-batch-highrecall # run batch pipeline in docker (high-recall)" \
	"" \
	"Overrides:" \
	"  make gui GUI_PYTHON=~/PycharmProjects/PythonProject/.venv1/bin/python" \
	"  make infer VIDEO=/path/to/video.avi OUTPUT_DIR=artifacts/custom_run" \
	"  make eval VIDEO=data/raw/input.avi ANNOTATIONS=data/annotations/manual_points.csv"

infer:
	$(PIPELINE_PYTHON) src/droplet_cv.py --video $(VIDEO) --output-dir $(OUTPUT_DIR) --mode balanced --skip-tuning --no-annotations

infer-highrecall:
	$(PIPELINE_PYTHON) src/droplet_cv.py --video $(VIDEO) --output-dir $(OUTPUT_DIR) --mode high-recall --skip-tuning --no-annotations

infer-baseline:
	$(PIPELINE_PYTHON) src/droplet_cv.py --video $(VIDEO) --output-dir $(OUTPUT_DIR) --mode baseline --skip-tuning --no-annotations

eval:
	$(PIPELINE_PYTHON) src/droplet_cv.py --video $(VIDEO) --annotations $(ANNOTATIONS) --output-dir $(OUTPUT_DIR) --mode balanced --skip-tuning

eval-highrecall:
	$(PIPELINE_PYTHON) src/droplet_cv.py --video $(VIDEO) --annotations $(ANNOTATIONS) --output-dir $(OUTPUT_DIR) --mode high-recall --skip-tuning

batch:
	$(PIPELINE_PYTHON) src/mlops_batch.py --mode balanced --output-dir $(BATCH_OUTPUT_DIR) --overwrite

batch-highrecall:
	$(PIPELINE_PYTHON) src/mlops_batch.py --mode high-recall --output-dir $(BATCH_OUTPUT_DIR) --overwrite

label:
	SYSTEM_VERSION_COMPAT=0 $(NAPARI_PYTHON) src/napari_labeling.py

gui:
	DROPLET_PIPELINE_PYTHON=$(PIPELINE_PYTHON) SYSTEM_VERSION_COMPAT=0 $(GUI_PYTHON) src/gui_app.py

test:
	$(PIPELINE_PYTHON) -m unittest discover -s tests -v

docker-build:
	docker compose build $(DOCKER_SERVICE)

docker-batch:
	docker compose run --rm $(DOCKER_SERVICE) --mode balanced

docker-batch-highrecall:
	docker compose run --rm $(DOCKER_SERVICE) --mode high-recall
