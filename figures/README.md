# Figures

Figures referenced by the paper ("TSKAN: Interpretable Machine Learning for
QoE modeling over Time Series Data", Kamal Singh et al.).

| File | Paper figure | Description |
|---|---|---|
| `kan_architecture.svg` | Fig. 2 / Fig. 4 | 1-layer KAN architecture: output is the sum of per-input trained activations |
| `fig2_kan_activations/M_*.svg` | Fig. 3 | Learned activation functions of the Netflix TSKAN for the magnitude (DC and first frequency) of each of the 6 variables |
| `fig2_kan_activations/phase_*.svg` | Fig. 3 | Learned activation functions of the Netflix TSKAN for the first-frequency phase of each variable (phase-sensitive features) |
| `fig3_phase_effects.svg` | Fig. 5 | Predicted MOS vs. time for shifted phase components, showing TSKAN's sensitivity to when a quality event occurs in the session |

Regenerate the activation plots (Fig. 3) by running
`experiments/01_tskan_netflix.ipynb` and re-plotting with
`model.plot(scale=4)` / `model.attribute(1, 0, plot=True)`.
