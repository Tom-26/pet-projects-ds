# Droplet CV Pipeline

Проект для подсчёта капель на видео, трекинга и оценки интенсивности свечения бактерий внутри капель.

## Что делает pipeline

1. Ищет кандидаты капель на bandpass-версии кадра.
2. Фильтрует ложные пики по локальной интенсивности.
3. Делает recovery-pass по соседним кадрам для слабых, но временно согласованных капель.
4. Подбирает параметры по размеченным кадрам.
5. Трекает капли между кадрами.
6. Считает интенсивность свечения и строит QC-артефакты.

## Структура проекта

```text
napari_marking/
├── README.md
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── pyproject.toml
├── requirements.inference.txt
├── venv/
├── mlops/
│   ├── README.md
│   ├── input/
│   │   ├── videos/
│   │   └── annotations/
│   └── output/
├── src/
│   ├── droplet_cv.py
│   ├── gui_app.py
│   ├── mlops_batch.py
│   └── napari_labeling.py
├── data/
│   ├── raw/
│   │   └── input.avi
│   ├── annotations/
│   │   ├── manual_points.csv
│   │   └── manual_points.npy
│   └── previews/
│       └── manual_points_preview.png
├── artifacts/
│   ├── tables/
│   ├── qc/
│   └── reports/
└── experiments/
    └── legacy/
        ├── denoise.py
        ├── hand_points.py
        ├── log_blob_detector.py
        └── smoothed_h=*.mp4
```

## Где что лежит

- `src/` - рабочий код, который надо поддерживать и улучшать.
- `mlops/input/videos/` - папка, куда удобно класть входные ролики для batch-запуска.
- `mlops/input/annotations/` - необязательная разметка для batch-оценки, имя файла должно совпадать с видео.
- `mlops/output/` - batch-результаты по каждому видео, плюс manifest и логи.
- `data/raw/` - входное видео.
- `data/annotations/` - ручная разметка для обучения и валидации.
- `data/previews/` - вспомогательные превью разметки.
- `artifacts/tables/` - CSV-результаты, метрики, таблицы детекций.
- `artifacts/qc/` - overlay-видео и PNG для ручной проверки качества.
- `artifacts/reports/` - итоговые JSON-отчёты по запуску.
- `experiments/legacy/` - старые черновики и промежуточные эксперименты, которые не участвуют в основном pipeline.

## Основные скрипты

- `src/droplet_cv.py` - основной pipeline detector + tracking + intensity + QC.
- `src/mlops_batch.py` - batch-обвязка: берёт все видео из входной папки и раскладывает результаты по отдельным output-папкам.
- `src/gui_app.py` - локальный desktop GUI для single-video и batch-запусков.
- `src/napari_labeling.py` - разметка точек в `napari`.
- `src/iteration1_observability.py` - сравнение preprocessing-веток detector с отчётными артефактами.
- `src/iteration2_ensemble.py` - эксперимент с ансамблем `baseline + background_subtracted`.
- `src/iteration3_rescoring.py` - rescoring merged candidates поверх `baseline + background_subtracted`.

## Запуск

Из папки `napari_marking`:

```bash
./venv/bin/python src/droplet_cv.py
```

Основной production-режим с лучшим балансом метрик:

```bash
./venv/bin/python src/droplet_cv.py --skip-tuning --mode balanced
```

High-recall режим, если важнее добрать максимум точек:

```bash
./venv/bin/python src/droplet_cv.py --skip-tuning --mode high-recall
```

Чистый baseline detector без ensemble/rescoring:

```bash
./venv/bin/python src/droplet_cv.py --skip-tuning --mode baseline
```

Запуск без любой локальной разметки, только inference:

```bash
./venv/bin/python src/droplet_cv.py --video path/to/video.avi --mode balanced --skip-tuning --no-annotations
```

Быстрый запуск без повторного тюнинга:

```bash
./venv/bin/python src/droplet_cv.py --skip-tuning
```

Запуск observability для iteration 1:

```bash
./venv/bin/python src/iteration1_observability.py
```

Запуск ансамбля для iteration 2:

```bash
./venv/bin/python src/iteration2_ensemble.py
```

Запуск rescoring для iteration 3:

```bash
./venv/bin/python src/iteration3_rescoring.py
```

Запуск разметки в napari:

```bash
./venv/bin/python src/napari_labeling.py
```

Минимальный batch/MLOps запуск:

```bash
./venv/bin/python src/mlops_batch.py --mode balanced
```

Docker-запуск:

```bash
docker compose run --rm droplet-cv --mode balanced
```

Быстрый запуск через `make`:

```bash
make help
make infer
make eval
make batch
make gui
make test
make docker-batch
```

GUI можно открыть и напрямую:

```bash
./venv/bin/python src/gui_app.py
```

Если `GUI` и основной pipeline живут в разных окружениях, это нормально:

```bash
make gui GUI_PYTHON=~/PycharmProjects/PythonProject/.venv1/bin/python
```

Внутри GUI сам CV-pipeline по умолчанию всё равно будет запускаться через `./venv/bin/python`, если он существует.

## Что лежит в artifacts

### `artifacts/tables/`

- `tuning_results.csv` - перебор параметров detector.
- `evaluation_per_frame.csv` - качество на размеченных кадрах.
- `detections_tracked.csv` - все детекции с треками и интенсивностью.
- `frame_summary.csv` - число капель и агрегаты по кадрам.
- `track_summary.csv` - агрегаты по трекам.
- `qc_problem_frames.csv` - рейтинг самых проблемных размеченных кадров.

### `artifacts/qc/`

- `qc_overlay.mp4` - overlay по всему видео.
- `preview_frame_0.png` - быстрый preview одного размеченного кадра.
- `qc_frame_rank_*.png` - top-k проблемных кадров с разметкой `TP/FP/FN`.

### `artifacts/reports/`

- `run_summary.json` - итог запуска, выбранные параметры и сводные метрики.

### `artifacts/experiments/iteration1/`

- `branch_metrics.csv` - сравнительная таблица метрик по веткам и split.
- `branch_best_params.csv` - лучшие параметры detector для каждой ветки.
- `holdout_metrics.svg` - диаграмма сравнения веток на holdout.
- `holdout_per_frame.csv` - покадровые ошибки по веткам на holdout.
- `split_summary.json` - использованный temporal split.
- `report.md` - короткий текстовый отчёт для документации.

### `artifacts/experiments/iteration2_ensemble/`

- `ensemble_metrics.csv` - сравнение baseline, bg-sub и двух режимов ансамбля.
- `holdout_per_frame.csv` - покадровые ошибки моделей на holdout.
- `ensemble_selective_detections.csv` - fused detections осторожного ансамбля.
- `ensemble_high_recall_detections.csv` - fused detections high-recall ансамбля.
- `ensemble_config.json` - параметры веток и правила fusion.
- `report.md` - короткая сводка результатов.

### `artifacts/experiments/iteration3_rescoring/`

- `rescoring_metrics.csv` - сравнение baseline, bg-sub и rescored union.
- `rescored_detections.csv` - итоговые детекции после rescoring.
- `rescored_all_per_frame.csv` - покадровые ошибки rescored model.
- `train_labeled_candidates.csv` - train-кандидаты с псевдоразметкой для обучения rescoring.
- `holdout_candidates_scored.csv` - holdout-кандидаты с вычисленным score.
- `rescoring_config.json` - признаки, веса модели и выбранный threshold.
- `report.md` - короткая сводка результатов.

## Почему detector сейчас устроен так

- `denoise` помогает локализации, но не должен напрямую определять биологическую интенсивность.
- Один глобальный порог по кадру даёт слишком много ложных точек.
- Поэтому detector сейчас не одношаговый, а состоит из трёх частей:
  - сначала пространственный поиск кандидатов,
  - потом фильтрация по локальной интенсивности.
  - затем temporal recovery для слабых кандидатов, которые согласованы с соседними кадрами.

## Режимы pipeline

### `balanced`

Основной рабочий режим для отчётности и повседневного анализа. Использует `baseline + background_subtracted` как источник кандидатов и rescoring поверх объединённого набора. Это лучший текущий компромисс между `precision` и `recall`.

Ориентир по holdout:

- `precision ≈ 0.904`
- `recall ≈ 0.846`
- `F1 ≈ 0.874`

### `high-recall`

Режим, если важнее не потерять слабые капли и собрать максимум точек. Использует более агрессивный union двух веток detector с дедупликацией. Обычно поднимает `recall`, но ценой большего числа `FP`.

Ориентир по holdout:

- `precision ≈ 0.816`
- `recall ≈ 0.900`
- `F1 ≈ 0.856`

### `baseline`

Режим для чистого сравнения с исходной веткой detector. Использует только `baseline_bandpass` и при наличии разметки может выполнять исходный подбор параметров без ансамбля и rescoring.

## Ветки detector

### `baseline_bandpass`

Основная рабочая ветка. Для каждого кадра строится `bandpass`-ответ:

- слабое сглаживание убирает пиксельный шум
- сильное сглаживание оценивает медленный фон
- их разность подчёркивает компактные яркие объекты, похожие на капли

Это текущий основной baseline, потому что он даёт лучший общий баланс между `precision`, `recall` и ошибкой подсчёта.

### `background_subtracted`

Ветка для более агрессивного подавления неравномерного фона:

- сначала оценивается локальный фон большим гауссовым размытием
- затем этот фон вычитается из исходного кадра
- после этого результат слегка сглаживается

Отличие от `baseline_bandpass` в том, что эта ветка сильнее борется с медленным перепадом яркости по полю зрения. Обычно она лучше добирает слабые капли и поднимает `recall`, но чаще даёт лишние ложные срабатывания и поэтому просаживает `precision`.

### `temporal_median_bandpass`

Ветка с временным сглаживанием:

- перед spatial detection каждый кадр заменяется медианой по короткому временному окну
- затем к этому temporally smoothed видео применяется обычный `bandpass`

Отличие от остальных в том, что она пытается подавлять шум не только по пространству, но и по времени. Такая ветка полезна, если шум случайный и быстро меняется от кадра к кадру. На текущих данных она оказалась слишком жёсткой и стала терять реальные слабые капли, поэтому сильно просел `recall`.

## Чем они отличаются коротко

- `baseline_bandpass`:
  лучший общий baseline и самый стабильный баланс метрик
- `background_subtracted`:
  лучше вытаскивает слабые капли на сложном фоне, но добавляет `FP`
- `temporal_median_bandpass`:
  лучше режет случайный шум, но в текущем виде переусредняет сигнал и теряет капли

## Полезные параметры

- текущие дефолты detector обновлены после расширения разметки до 30 кадров
- `--mode balanced|high-recall|baseline` - готовые режимы запуска под разные приоритеты качества.
- `--no-annotations` - полностью отключить загрузку локальной CSV-разметки и запустить чистый inference.
- `--selection-metric f1|count_mae|balanced` - чем выбирать лучший detector.
- `--thresholds` и `--min-distances` - spatial-часть detector.
- `--net-p90-thresholds` и `--net-mean-thresholds` - второй каскад фильтрации.
- `--track-max-distance` и `--track-max-gap` - параметры трекера.
- `--disable-recovery-pass` - выключить temporal recovery и оставить только сильный detector.
- `--qc-top-k` - сколько проблемных кадров сохранить PNG.
- `--intensity-low-threshold` и `--intensity-high-threshold` - ручная калибровка классов свечения.

## Minimal MLOps

### Входы

- положить видео в `mlops/input/videos/`
- при желании положить CSV-разметку в `mlops/input/annotations/`
- имя CSV должно совпадать с именем видео, например `sample_01.avi` и `sample_01.csv`
- если CSV нет, batch-runner сам включает `--no-annotations`, то есть считает только inference без метрик

### Выходы

Для каждого видео создаётся отдельная папка:

- `mlops/output/<video_stem>/tables`
- `mlops/output/<video_stem>/qc`
- `mlops/output/<video_stem>/reports`

Также batch-runner пишет:

- `mlops/output/manifest.json` - общий список запусков и краткие метрики
- `mlops/output/<video_stem>.stdout.log` - stdout конкретного job
- `mlops/output/<video_stem>.stderr.log` - stderr конкретного job

### Docker

Контейнер собирается из `Dockerfile`, а `docker-compose.yml` уже монтирует удобные volume:

- `./mlops/input:/app/mlops/input`
- `./mlops/output:/app/mlops/output`

То есть пользовательский сценарий простой:

1. Положить видео в `mlops/input/videos/`.
2. Запустить `docker compose run --rm droplet-cv --mode balanced`.
3. Забрать результаты из `mlops/output/`.

## Makefile shortcuts

- `make infer` - быстрый balanced inference без аннотаций
- `make infer-highrecall` - быстрый high-recall inference без аннотаций
- `make infer-baseline` - быстрый baseline inference без аннотаций
- `make eval` - balanced запуск с локальной CSV-разметкой
- `make eval-highrecall` - high-recall запуск с локальной CSV-разметкой
- `make batch` - batch-runner по `mlops/input/videos`
- `make batch-highrecall` - batch-runner в high-recall режиме
- `make label` - открыть `napari` для разметки
- `make gui` - открыть desktop GUI
- `make test` - запустить unit tests
- `make docker-build` - собрать контейнер
- `make docker-batch` - batch-run в Docker
- `make docker-batch-highrecall` - high-recall batch-run в Docker

Можно переопределять пути прямо в команде:

```bash
make infer VIDEO=/path/to/video.avi OUTPUT_DIR=artifacts/custom_run
make eval VIDEO=data/raw/input.avi ANNOTATIONS=data/annotations/manual_points.csv
make gui GUI_PYTHON=~/PycharmProjects/PythonProject/.venv1/bin/python
```

## Tests

Unit-тесты лежат в `tests/` и покрывают быстрые deterministic-части pipeline:

- split и разбор разметки
- preprocessing utility functions
- merging / high-recall fusion / temporal support
- parameter selection
- intensity classification и summary tables
- batch helper functions из `mlops_batch.py`

Также есть лёгкие integration tests на synthetic video:

- прямой CLI-прогон `droplet_cv.py`
- batch CLI-прогон `mlops_batch.py`

Запуск:

```bash
make test
```

## CI/CD

В репозитории уже добавлены workflow:

- `.github/workflows/ci.yml` - запускает `py_compile`, `unittest` и проверку `docker compose config`
- `.github/workflows/docker-publish.yml` - собирает Docker image и публикует его в `GHCR`

Что делает `CI`:

- стартует на `push`, `pull_request` и `workflow_dispatch`
- использует `Python 3.11`
- ставит зависимости из `requirements.inference.txt`
- гоняет весь тестовый набор из `tests/`

Что делает `Docker Publish`:

- стартует на `push` в `main`, на теги `v*` и вручную
- публикует образ в `ghcr.io/<owner>/<repo>`
- добавляет теги branch/tag/sha и `latest` для default branch

Минимальная настройка в GitHub:

1. Залить проект в GitHub-репозиторий.
2. Убедиться, что default branch называется `main` или поправить workflow под своё имя ветки.
3. В `Settings -> Actions -> General` разрешить GitHub Actions.
4. В `Settings -> Actions -> General -> Workflow permissions` включить `Read and write permissions`, если хочешь, чтобы publish в `GHCR` работал без дополнительных ограничений.
5. Первый push в `main` автоматически запустит `CI` и `Docker Publish`.

Проверка локально перед push:

```bash
make test
docker compose config
```

## GUI

`gui_app.py` даёт две вкладки:

- `Single Video` - запуск одного ролика с выбором `mode`, optional CSV-аннотаций и своей output-папки
- `Batch` - запуск по папке `mlops/input/videos` или любой другой директории с роликами

После прогона GUI показывает:

- краткую сводку по последнему run или manifest
- live log процесса
- встроенный `Output Browser` с деревом артефактов
- отдельную `QC Gallery` для проблемных кадров
- вкладку `Run History` с прошлыми single/batch запусками
- цветовую индикацию в `Run History` для хороших, средних и проблемных прогонов
- ручные пороги в `Run History`, чтобы подстроить раскраску под свою отчётность
- предпросмотр `PNG`, `JSON`, `CSV`, `log`, `md`
- попытку встроенного просмотра `qc_overlay.mp4`, если в Qt-сборке доступно multimedia
- навигацию по QC-кадрам `Prev/Next` и метаданные `TP/FP/FN`, если найден `qc_problem_frames.csv`
- кнопки `Reopen Selected Run`, `Load Browser`, `Load QC` и `Open Output Folder` для выбранного запуска
- быстрый переход к output-папке через системный файловый менеджер

GUI теперь использует `PyQt5`, а не `tkinter`, чтобы не упираться в известные macOS/Tk проблемы. При этом сам GUI может быть запущен из одного окружения, а вычислительный pipeline из другого.

## Следующие улучшения

1. Поднять recall на проблемных кадрах из `artifacts/tables/qc_problem_frames.csv`.
2. Добавить ещё размеченные кадры для более устойчивого тюнинга.
3. При необходимости перейти к более сильному tracker с моделью скорости.
4. Калибровать классы свечения по реальным биологическим меткам, а не по квантилям.
