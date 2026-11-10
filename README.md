# TSKAN: Interpretable Machine Learning for QoE modeling over Time Series Data

Source code for the paper

> **TSKAN: Interpretable Machine Learning for QoE modeling over Time Series Data**
> Kamal Singh, Priyanka Rawat, Sami Marouani, Baptiste Jeudy,
> Université Jean Monnet, Saint-Étienne, CNRS, Lab. Hubert Curien / Telecom Saint-Étienne, France

```bibtex
@article{singh2025tskan,
  title   = {TSKAN: Interpretable Machine Learning for QoE modeling over Time Series Data},
  author  = {Singh, Kamal and Rawat, Priyanka and Marouani, Sami and Jeudy, Baptiste},
  year    = {2025}
}
```

## What is TSKAN?

TSKAN models Quality of Experience (QoE) from raw video-streaming time series.
It combines

1. **compact frequency-domain (DFT) features** — DC magnitude plus the
   magnitude and phase of the first frequency, per variable (F = 1 in the
   paper), and
2. a **1-layer Kolmogorov-Arnold Network (KAN)** as an interpretable readout,
   with top-k selection of the most important features via the KAN input
   importance scores.

The result is a small (120-parameter) model whose learned univariate
activation functions are directly visualizable (see `figures/`), while
approaching performance of much larger black-box time series models.

## Repository layout

```
tskan/                  python package: data loaders, DFT features, TSKAN model
scripts/                data preparation from raw session logs
  prepare_netflix.py    LIVE-Netflix-II pkls  -> data/netflix2/*.ts
  prepare_waterloo.py   Waterloo 3 session CSVs -> data/waterloo3/*.ts
experiments/
  01_tskan_netflix.ipynb          TSKAN on LIVE-Netflix-II        (paper Secs. III-V)
  02_tskan_waterloo3.ipynb        TSKAN on Waterloo QoE Database 3
  03_baselines_deep_learning.ipynb  deep time series baselines    (paper Table I)
  04_baselines_interpretable.ipynb  interpretable baselines       (paper Table II)
data/                   NOT included here — see "Datasets" below
figures/                figures of the paper (KAN activations, phase effects)
results/                results tables of the paper
```

## Datasets

| Dataset | Sequence | Splits (70/15/15) |
|---|---|---|
| LIVE-Netflix-II [1] | 16 chunks x 7 variables | 282 / 54 / 56 |
| Waterloo QoE Database 3 [2] | 5 chunks x 7 variables | 326 / 65 / 59 |

Variables (per chunk): rebuffering duration, video bitrate, chunk duration,
QP, frame rate, width, height. Target: z-scored MOS.

- TSKAN (notebooks 01/02) uses the first 6 variables for both datasets
  (height is dropped; it is constant within a session).
- The deep baselines (notebook 03) use all 7 variables for LIVE-Netflix-II
  (height varies across representations) and 6 for Waterloo 3 — which is why
  Table I lists two parameter counts per architecture.

### Getting the data

The data (raw and processed) is **not included in this repository**: it can be downloaded from their respective sources (check references).

Its layout is:

```
netflix2/
  netflixpluspadded_{TRAIN,VALID,TEST}.ts     TSKAN inputs: 7 vars x 16 chunks + z-scored MOS
  compnetflixpluspadded_{TRAIN,VALID,TEST}.ts expert-crafted features for interpretable baselines
waterloo3/
  waterloo3_zscore_{TRAIN,VALID,TEST}.ts      TSKAN inputs: 7 vars x 5 chunks + z-scored MOS
  compwaterloo_{TRAIN,VALID,TEST}.ts          expert-crafted features for interpretable baselines
  raw/                                        raw Waterloo session logs (might not be allowed to redistribute)
```

To run the experiments, place the datasets in the folder tskan-data under `data/`

`scripts/` regenerates the processed splits byte-for-byte from the raw data.

## Installation

Python 3.12 is recommended (the paper was run on 3.12).

```bash
conda env create -f environment.yml     # or: python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

A GPU (CUDA) is useful for `experiments/03_baselines_deep_learning.ipynb`;
the TSKAN notebooks and the interpretable baselines run fine on CPU.

## Quickstart

Run notebooks from the repository root:

```bash
python -m jupyter notebook experiments/01_tskan_netflix.ipynb
```

1. `01_tskan_netflix.ipynb` / `02_tskan_waterloo3.ipynb` — train TSKAN
   (two stages: KAN on all DFT features, then top-k selection and a KAN on
   the top-10 features) and report the test RMSE.
2. `03_baselines_deep_learning.ipynb` — re-run the deep baselines (Table I).
   The full grid (23 architectures x 4 learning rates x 20 restarts x 200
   epochs) is expensive; see the notebook for the protocol.
3. `04_baselines_interpretable.ipynb` — re-run the interpretable baselines
   (Table II).

## Results

Test RMSE (z-scored MOS) reported in the paper (Tables I and II):

| Model | LIVE-Netflix-II | Waterloo 3 | Params (netflix / waterloo) |
|---|---|---|---|
| Transformer | 0.2051 | 0.4274 | 3,350,017 / 3,347,201 |
| LSTM-FCN | 0.2046 | 0.4329 | 315,973 / 311,973 |
| FCN | 0.2066 | 0.4152 | 269,569 / 268,673 |
| ResNet | 0.2017 | 0.4986 | 480,897 / 481,409 |
| InceptionTime | 0.2116 | 0.4904 | 456,129 / 455,937 |
| LSTM bi-dir. | 0.1850 | 0.4107 | 328,201 / 328,201 |
| LSTM uni-dir. | 0.1865 | 0.3889 | 124,101 / 204,901 |
| **TSKAN (this work)** | **0.2169** | **0.4716** | **120** |
| Decision Tree (d=5) | 0.6020 | 0.5909 | depth 5 |
| Linear Regression | 0.5092 | 0.6695 | 6 |
| LASSO | 0.5461 | 0.6587 | 6 |
| EBM | 0.4914 | 0.4722 | 7,875 |
| NAM | 0.5480 | 0.5748 | 11,361 |

Some numbers are non-deterministic across hardware/runs; expect
small deviations.

### Reproducing / verification notes

Re-running the pipelines in this repo (Python 3.12, `tsai==0.4.0`,
`pykan==0.2.8`, CPU LBFGS for TSKAN) gives:

| Model | Paper | Re-run (netflix / waterloo) |
|---|---|---|
| Transformer | 0.2051 / 0.4274 | 0.2338 / 0.3859 |
| LSTM-FCN | 0.2046 / 0.4329 | 0.1854 / 0.4188 |
| FCN | 0.2066 / 0.4152 | 0.2025 / 0.4258 |
| ResNet | 0.2017 / 0.4986 | 0.1891 / 0.5540 |
| InceptionTime | 0.2116 / 0.4904 | 0.1985 / 0.4659 |
| LSTM bi-dir. | 0.1850 / 0.4107 | 0.1854 / 0.3888 |
| LSTM uni-dir. | 0.1865 / 0.3889 | 0.1985 / 0.3882 |
| TSKAN | 0.2169 / 0.4716 | 0.2131–0.2252 / 0.4784–0.4817 |
| Decision Tree | 0.6020 / 0.5909 | 0.6020 / 0.5909 |
| Linear Regression | 0.5092 / 0.6695 | 0.5092 / 0.6695 |
| LASSO | 0.5461 / 0.6587 | 0.5112 / 0.6779 (test set; the paper reported the *validation* RMSE for LASSO) |
| EBM | 0.4914 / 0.4722 | 0.4914 / 0.4722 |
| NAM | 0.5480 / 0.5748 | 0.5480 / 0.5748 |

- The re-run netflix top-10 feature set is identical to the paper's
  (`M_rebuffering_duration(0/1)`, `M_video_bitrate(0)`, `M_chunk_duration(0/1)`,
  `phase_chunk_duration(1)`, `M_qp(0)`, `phase_qp(1)`, `M_framerate(0)`,
  `M_width(0)`).
- NAM parameter counts depend on the `dnamite` build (`>=1.0` trains an
  ensemble of five models; the paper used an earlier single-model build).
- TSKAN runs are LBFGS from a fixed seed; expect ~1-2% RMSE variation.
- Deep baselines were re-run as a full grid (23 architectures x 4 learning
  rates x 20 restarts x 200 epochs); the best configuration per architecture
  (by validation RMSE) is shown above, matching the paper's selection
  protocol. `xresnet1d34` (fastai>=2.7.19 API incompatibility) and `mWDN`
  on Waterloo (series too short for its downsampling) are skipped
  automatically with a message. Full runs: `experiments/results_dl_*.csv`.
- `scripts/prepare_*.py` regenerate the processed `.ts`/CSV splits
  byte-for-byte from the raw data (see "Getting the data").

## References

[1] S. Bampis, N. P. Chouliaras, I. Leontiadis, V. A. Leventis, C. N.
    Vasilakos, and A. C. Bejanin, "An in-depth analysis of the LIVE
    QoE database of adaptive streaming," 2018.

[2] A. Duanmu, C. G. S. Shen, X. Lin, and D. R. Bull, "Waterloo QoE
    database 3: A new dataset of subjective video quality evaluations,"
    in Proc. SPIE, 2018.

[3] Z. Liu, L. Wang, S. V. Ermon, and M. Z. Ghorbani, "Kan: Kolmogorov-Arnold
    networks," 2024. https://github.com/KindXiaoming/pykan

## License

This code is licensed under the [CeCILL-B](LICENSE) license
(Free Software license issued by CEA/CNRS/INRIA, France; OSI-approved,
permissive — similar in spirit to the BSD licenses).
