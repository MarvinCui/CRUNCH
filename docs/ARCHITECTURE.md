# CRUNCH architecture

CRUNCH is the team's original dual-branch method plus second-order harmonics. The original architecture and this extension are both team contributions. The authoritative numerical implementation is `original_core.py`, `event_features.py` and `predict.py`.

## Input and observation support

For each field-year, use finite NDVI, LSWI and NDTI observations at actual dates. Index definitions are `(NIR−RED)/(NIR+RED)`, `(NIR−SWIR1)/(NIR+SWIR1)`, and `(SWIR1−SWIR2)/(SWIR1+SWIR2)`. The release starts from frozen field-level index summaries, not raw image bands.

Keep May–November observations. Require at least 5 valid dates, at least 2 in June–August, at least 2 in September–November, and a midsummer smoothed NDVI maximum ≥ 0.4. Failing this yields `uncertain`. Smoothing is a centered three-observation rolling median, with at least two observations. No regular time-grid interpolation is used by CRUNCH.

## Original feature branches

**Event branch, 9 features:** `ndvi_pre`, `ndvi_pre_fraction_of_peak`, `ndvi_drydown_per_day`, `ndvi_step`, `lswi_pre`, `lswi_step`, `ndti_post`, `ndti_step`, `peak_to_harvest_days`.

The event extractor requires at least 7 observations. After its full-season smoothed NDVI peak, candidate declines must have a first gap > 0 and ≤ 18 days, following gap ≤ 24 days, NDVI decrease ≥ 0.09, and a persistent decrease ≥ 0.07 over the following two observations. Candidates are ranked by `(persistence + decrease) / sqrt(first_gap)`. Python's tuple tie-breaking is retained. The estimated event lies between the two observations; no ground-truth harvest date enters the model.

Pre-event summaries use the preceding 14 days through the last pre-event observation; post-event summaries use the first post-event observation through 12 days afterward. Drydown is the NDVI regression slope over the preceding 28 days, requiring at least 3 dates spanning 8 days. NDTI post-event uses the first post-event observation. The event interval midpoint determines peak-to-event duration. The optional HPI code path is inactive in CRUNCH.

**Peak-aligned branch, 18 features:** for offsets `0,15,30,45,60,75` days, take NDVI/LSWI/NDTI medians within ±9 days. This branch anchors to the last June–August date reaching at least 95% of the midsummer smoothed NDVI maximum. Its anchor differs from the full-season peak used by the event extractor. Empty windows are missing values.

## Harmonic representation

For each index, fit the actual observed values against

```text
A(t) = [1, cos(ωt), sin(ωt), cos(2ωt), sin(2ωt)]
ω = 2π / 365.25
t = days since January 1 of the field's year

β = argminβ ||Aβ − x||² / n + 0.01 ||β[1:]||²
```

The intercept is unpenalized. Solve the ridge normal equations independently for each field-year and index. Feature order: NDVI, LSWI, NDTI, each with `intercept,cos1,sin1,cos2,sin2`. The 15 coefficients are appended to each branch: **24 event inputs**, **33 peak-aligned inputs**. Harmonics are calculated per sample without labels or pooled target-domain fitting. These are regularized seasonal summaries, not exact periodicity estimates from a full observed year.

## Training and routing

Each branch trains on source rows having at least one nonmissing original feature in its support list. In this release the event branch trains on 353 WA examples; the peak-aligned branch trains on all 599. Within each branch, weights are inverse frequency of `(region,label)`, normalized to mean 1.

Pipeline: `SimpleImputer(strategy='mean', keep_empty_features=True)` → `StandardScaler` fitted with those weights → L2 `LogisticRegression(C=0.3,max_iter=1000)` fitted with those weights. Imputation means are unweighted; scaling and classifier fitting are weighted. Labels encode grain=0, silage=1. All learned preprocessing is source-only.

At inference, after coverage checks, use the event branch when its original support features are present; otherwise use the peak-aligned branch. Harmonics do not independently determine eligibility. Remaining unsupported rows are uncertain. A score ≥ **0.5** predicts silage. No threshold optimization, averaging, weather or foundation-model features are part of CRUNCH 1.0.0.

The branch logit can be decomposed exactly into its intercept plus standardized features multiplied by logistic coefficients. These are model explanations, not causal effects of agricultural management.
