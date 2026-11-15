# CRUNCH comparison

All models were trained on 599 WA Yakima field-years and evaluated on 44 MD BARC competition field-years from 34 physical fields. All used the same frozen NDVI, LSWI and NDTI observations.

| Model | Configuration | Threshold | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| CRUNCH | H2, α = 0.01, C = 0.3 | 0.5 | 95.45% (42/44) | 0.939 |
| MultiRocket [1] | Class-balanced | 0.680131855 | 87.88% ± 4.73 pp | 0.816 ± 0.086 |
| ROCKET [2] | Class-balanced | 0.5771088 | 73.48% ± 1.31 pp | 0.719 ± 0.012 |
| TempCNN [3] | Class-balanced | 0.999923275 | 76.52% ± 23.21 pp | 0.732 ± 0.210 |

For each method, the table shows the previously evaluated configuration with the highest mean MD Macro-F1, breaking ties by Accuracy. Candidates include the fixed 0.5 threshold and the previously evaluated threshold selected on NL. This is a retrospective comparison on MD. Each published method uses one configuration across seeds 17, 37 and 66. ± denotes sample standard deviation across seeds. CRUNCH is the deterministic release checkpoint.

ROCKET and MultiRocket use the multivariate aeon 1.6.0 implementations. TempCNN uses a PyTorch port. The published methods use 32 interpolated time steps per channel. CRUNCH uses event features and harmonics fitted at observation dates.

## References

1. Tan, C. W., Dempster, A., Bergmeir, C., & Webb, G. I. (2022). MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification. Data Mining and Knowledge Discovery, 36, 1623–1646. [DOI](https://doi.org/10.1007/s10618-022-00844-1)
2. Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. Data Mining and Knowledge Discovery, 34, 1454–1495. [DOI](https://doi.org/10.1007/s10618-020-00701-z)
3. Pelletier, C., Webb, G. I., & Petitjean, F. (2019). Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series. Remote Sensing, 11(5), 523. [DOI](https://doi.org/10.3390/rs11050523)
