# Iteration 3 Rescoring

- selected threshold: `0.810`
- train fitted candidates: `1387`
- holdout candidates scored: `1596`

## Holdout results

| Model | Precision | Recall | F1 | Count MAE |
| --- | ---: | ---: | ---: | ---: |
| baseline_bandpass | 0.886 | 0.849 | 0.867 | 6.50 |
| background_subtracted | 0.816 | 0.888 | 0.850 | 10.94 |
| rescored_union | 0.904 | 0.846 | 0.874 | 7.62 |