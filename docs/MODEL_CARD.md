# CRUNCH model card

- **Name:** CRUNCH — Corn End-Use Recognition Using Normalized Indices, Change Points, and Harmonics.
- **Authors:** TIANYANG LIU and BORAN CUI.
- **Version:** 1.0.0. **Task:** retrospective, end-of-season grain/silage corn recognition at field-year level.
- **Checkpoint:** `models/crunch.joblib`. **Implementation:** two routed L2 logistic regression pipelines with original event/peak features and 15 harmonic coefficients.
- **License:** MIT for team implementation and model weights; source-specific data status in `DATA_LICENSE.md`.

## Intended use

Research on corn end-use classification from adequately observed May–November NDVI/LSWI/NDTI trajectories. Input fields should already be identified as corn. The model does not detect corn versus other crops, ingest raw imagery, estimate yield, or provide validated early-season predictions.

## Training and evaluation

599 WA field-years; MD 44 field-years held out from fitting. Fixed order 2, harmonic α=0.01, classifier C=0.3, threshold 0.5. No NL threshold selection or external foundation-model features. MD Accuracy 42/44, Macro-F1 0.939394, balanced accuracy 0.916667, AUC 0.953125. Confusion matrix, rows=true grain/silage and columns=predicted grain/silage:

```text
[[32, 0],
 [ 2,10]]
```

All 44 test rows are scored. Unsupported new inputs yield uncertain. Scores have not been calibrated for deployment populations. Frozen probabilities and routes are in `reference/test_predictions.csv`.

## Limitations

MD was repeatedly examined during historical selection; the fixed release is exploratory rather than an untouched blind-validation result. Only one WA training year and a small MD test population are represented. Class proportions, sensor responses, observation density and seasonal timing differ by region. Model performance outside this population is unresolved. Early-season operation and severe cloud gaps have not been validated. Event dates are spectrally inferred intervals, not ground-truth harvest dates.

The benchmark documents both default and class-balanced adaptations of published models. Their weaker fixed-threshold Accuracy does not establish universally weaker representations: MiniRocket has higher MD AUC, and an exploratory NL-selected threshold substantially improves its classification scores. See [BENCHMARKS.md](BENCHMARKS.md).

Training and verification run on CPU. The release refit is checked against frozen predictions, not required to yield byte-identical serialized files across library versions. `reference/validation.json` records numerical tolerance and measured errors.
