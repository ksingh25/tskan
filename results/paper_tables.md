# Results (paper)

Test RMSE on z-scored MOS.

## Table I — Deep learning models

| Model | Parameters (netflix / waterloo) | LIVE-Netflix-II RMSE | Waterloo III RMSE |
|---|---|---|---|
| Transformer | 3,350,017 / 3,347,201 | 0.2051 | 0.4274 |
| LSTM-FCN | 315,973 / 311,973 | 0.2046 | 0.4329 |
| FCN | 269,569 / 268,673 | 0.2066 | 0.4152 |
| ResNet | 480,897 / 481,409 | 0.2017 | 0.4986 |
| InceptionTime | 456,129 / 455,937 | 0.2116 | 0.4904 |
| LSTM bi-dir. | 328,201 / 328,201 | **0.1850** | 0.4107 |
| LSTM uni-dir. | 124,101 / 204,901 | 0.1865 | **0.3889** |

Reproduce with `experiments/03_baselines_deep_learning.ipynb`.

## Table II — Interpretable models

| Model | Parameters | LIVE-Netflix-II RMSE | Waterloo III RMSE |
|---|---|---|---|
| TSKAN | 120 | **0.2169** | 0.4716 |
| DT (as in [22]) | depth = 5 | 0.6020 | 0.5909 |
| LR | 6 | 0.5092 | 0.6695 |
| LASSO | 6 | 0.5461 | 0.6587 |
| EBM (as in [22]) | 7,875 | 0.4914 | 0.4722 |
| NAM (as in [22]) | 11,361 | 0.5480 | 0.5748 |

TSKAN: trained with F = 1 (components 0 and 1) and top-k = 10.
Reproduce with `experiments/01_tskan_netflix.ipynb`,
`experiments/02_tskan_waterloo3.ipynb` and `experiments/04_baselines_interpretable.ipynb`.

