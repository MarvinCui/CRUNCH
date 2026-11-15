# Published-method comparisons

## Protocol

All results below use **WA599 for fitting and MD BARC44 for evaluation**, with the same frozen NDVI/LSWI/NDTI observations. All decision thresholds are **0.5**. Sequence methods interpolate May 1–November 30 to **3 channels × 32 time steps**, with constant edge filling, then standardize channels using WA only. CRUNCH uses its native event and irregular-date harmonic representation.

Both default and class-balanced training are shown for each published method; **no MD-based mode or seed selection** is performed. Results are the mean ± sample standard deviation over seeds **17, 37, 66**. CRUNCH is the frozen deterministic release. Standard deviations describe seed variation, not uncertainty from sampling new farms. This fixed-threshold comparison does not require NL data.

| Method | Training | Accuracy (%) | Macro-F1 |
|---|---|---:|---:|
| **CRUNCH** | Fixed released model | **95.45** | **0.939** |
| ROCKET [1] | Default | 31.82 ± 2.27 | 0.281 ± 0.032 |
| ROCKET [1] | Class-balanced | 47.73 ± 4.55 | 0.475 ± 0.047 |
| MiniRocket [2] | Default | 34.09 ± 2.27 | 0.312 ± 0.030 |
| MiniRocket [2] | Class-balanced | 45.45 ± 6.01 | 0.448 ± 0.068 |
| MultiRocket [3] | Default | 31.82 ± 2.27 | 0.281 ± 0.032 |
| MultiRocket [3] | Class-balanced | 33.33 ± 1.31 | 0.302 ± 0.018 |
| TempCNN [4] | Default | 54.55 ± 23.62 | 0.519 ± 0.264 |
| TempCNN [4] | Class-balanced | 30.30 ± 2.62 | 0.259 ± 0.039 |

The team's original architecture without harmonics scores Accuracy 90.91%, Macro-F1 0.870588 on this split. It is our own ablation, not an external paper baseline.

## Implementation and reproducibility

- **ROCKET / MiniRocket / MultiRocket:** aeon 1.6.0 multivariate implementations, `n_kernels=10000`, source-only `RidgeClassifierCV` with 10 alphas log-spaced from 10⁻³ to 10³, and default versus balanced class weights. Actual representation dimensions are 20,000 / 9,996 / 79,968. The ridge decision score is passed through sigmoid; 0.5 is therefore decision value zero, **not a calibrated 50% probability**. ROCKET retains aeon's per-instance normalization default.
- **TempCNN:** local PyTorch architecture port of the public 2019 structure: three Conv1D layers with 64 channels and width 5, then a 256-unit dense layer; BN/ReLU/dropout 0.5, two-class output, kernel-only L2 10⁻⁶, Adam learning rate 0.001, batch 32. A fixed source-only physical-field-disjoint 569/30 split chooses epochs with maximum 20 and patience 0, followed by a fresh all-599 refit. The internal validation set contains only 2 grain examples; all six runs selected 3 epochs. This small split and the port are material limitations. Hidden initialization is PyTorch Kaiming-normal; it differs slightly from Keras He-normal. GPU TF32 is disabled.
- **CRUNCH:** fixed H2, α=0.01, C=0.3, threshold 0.5; train-only imputation and scaling. The event branch fits 353 source examples and fallback fits all 599. No test labels enter branch routing.

`benchmarks/reference/` includes all 24 runs' MD predictions and their metrics, not only favorable runs. `benchmarks/README.md` provides scripts for rerunning the four adapted methods using the released data. The library implementations, input channels/length, preprocessing and TempCNN framework differ from the original papers' datasets and execution environments. These numbers are **adapted-method results on our dataset**, not the original papers' reported scores.

## What this comparison does and does not show

CRUNCH has the highest Accuracy and Macro-F1 among these implementations at fixed threshold 0.5. However, equal numerical thresholds do not imply equally calibrated scores across models. For example, balanced MiniRocket has MD **AUC 0.986**, compared with CRUNCH's **0.953**. Its low default-threshold Accuracy does not imply weak ranking ability.

In a separately reported, post-hoc diagnostic, using NL432 only to select weighting and a shared threshold across seeds raised MiniRocket MD Accuracy to **93.18% ± 2.27%**, Macro-F1 **0.912 ± 0.025**. Applying the same NL threshold rule to the fixed CRUNCH model gave **88.64% / 0.832**. That diagnostic is not part of this release's fixed model or the table above, and its NL data are not included here. It changes the ranking, so the fixed-threshold table must not be used to claim universal superiority over the paper methods.

MD has only 44 field-years from 34 physical fields and was repeatedly inspected during historical development. CRUNCH had a different historical search budget from the baseline adaptations. No new blind-test, comprehensive state-of-the-art, equal-search-budget, or broad US-wide superiority claim is supported. CPU and GPU timings are also not directly comparable.

## References

1. **Dempster, A., Petitjean, F., and Webb, G. I. (2020).** *ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels.* Data Mining and Knowledge Discovery, 34, 1454–1495. [Paper](https://doi.org/10.1007/s10618-020-00701-z) · [Authors' code](https://github.com/angus924/rocket).
2. **Dempster, A., Schmidt, D. F., and Webb, G. I. (2021).** *MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification.* Proceedings of KDD 2021. [Paper](https://doi.org/10.1145/3447548.3467231) · [Authors' code](https://github.com/angus924/minirocket).
3. **Tan, C. W., Dempster, A., Bergmeir, C., and Webb, G. I. (2022).** *MultiRocket: multiple pooling operators and transformations for fast and effective time series classification.* Data Mining and Knowledge Discovery, 36, 1623–1646. [Paper](https://doi.org/10.1007/s10618-022-00844-1) · [Authors' code](https://github.com/ChangWeiTan/MultiRocket).
4. **Pelletier, C., Webb, G. I., and Petitjean, F. (2019).** *Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series.* Remote Sensing, 11(5), 523. [Paper](https://doi.org/10.3390/rs11050523) · [Authors' code](https://github.com/charlotte-pel/temporalCNN).

Machine-readable citations: [references.bib](references.bib). CRUNCH authorship and software citation: [CITATION.cff](../CITATION.cff).
