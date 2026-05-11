# Iteration 1 Detector Observability

## Temporal split

- blocks: `[[0, 1, 2, 3, 4, 5], [8, 9, 10, 11, 12], [20], [30], [40], [48, 49, 50, 51, 52, 53], [60], [70], [78, 79, 80, 81, 82], [86, 87, 88]]`
- train frames: `[0, 1, 2, 3, 4, 5, 20, 40, 60, 78, 79, 80, 81, 82]`
- holdout frames: `[8, 9, 10, 11, 12, 30, 48, 49, 50, 51, 52, 53, 70, 86, 87, 88]`

## Holdout summary

| Branch | Precision | Recall | F1 | Count MAE |
| --- | ---: | ---: | ---: | ---: |
| Baseline bandpass | 0.886 | 0.849 | 0.867 | 6.50 |
| Local background subtraction | 0.816 | 0.888 | 0.850 | 10.94 |
| Temporal median + bandpass | 0.923 | 0.433 | 0.589 | 48.75 |

## Best params

| Branch | small_sigma | big_sigma | threshold_abs | min_distance | min_net_intensity_p90 | min_net_intensity_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline bandpass | 1.2 | 11.0 | 20.5 | 5 | 31.0 | 9.0 |
| Local background subtraction | 1.2 | 9.0 | 20.0 | 5 | 27.0 | 9.0 |
| Temporal median + bandpass | 1.0 | 11.0 | 19.5 | 5 | 27.0 | 9.0 |