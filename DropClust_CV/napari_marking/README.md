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
├── pyproject.toml
├── venv/
├── src/
│   ├── droplet_cv.py
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
- `data/raw/` - входное видео.
- `data/annotations/` - ручная разметка для обучения и валидации.
- `data/previews/` - вспомогательные превью разметки.
- `artifacts/tables/` - CSV-результаты, метрики, таблицы детекций.
- `artifacts/qc/` - overlay-видео и PNG для ручной проверки качества.
- `artifacts/reports/` - итоговые JSON-отчёты по запуску.
- `experiments/legacy/` - старые черновики и промежуточные эксперименты, которые не участвуют в основном pipeline.

## Основные скрипты

- `src/droplet_cv.py` - основной pipeline detector + tracking + intensity + QC.
- `src/napari_labeling.py` - разметка точек в `napari`.

## Запуск

Из папки `napari_marking`:

```bash
./venv/bin/python src/droplet_cv.py
```

Быстрый запуск без повторного тюнинга:

```bash
./venv/bin/python src/droplet_cv.py --skip-tuning
```

Запуск разметки в napari:

```bash
./venv/bin/python src/napari_labeling.py
```

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

## Почему detector сейчас устроен так

- `denoise` помогает локализации, но не должен напрямую определять биологическую интенсивность.
- Один глобальный порог по кадру даёт слишком много ложных точек.
- Поэтому detector сейчас не одношаговый, а состоит из трёх частей:
  - сначала пространственный поиск кандидатов,
  - потом фильтрация по локальной интенсивности.
  - затем temporal recovery для слабых кандидатов, которые согласованы с соседними кадрами.

## Полезные параметры

- `--selection-metric f1|count_mae|balanced` - чем выбирать лучший detector.
- `--thresholds` и `--min-distances` - spatial-часть detector.
- `--net-p90-thresholds` и `--net-mean-thresholds` - второй каскад фильтрации.
- `--track-max-distance` и `--track-max-gap` - параметры трекера.
- `--disable-recovery-pass` - выключить temporal recovery и оставить только сильный detector.
- `--qc-top-k` - сколько проблемных кадров сохранить PNG.
- `--intensity-low-threshold` и `--intensity-high-threshold` - ручная калибровка классов свечения.
