# SPDX-License-Identifier: MIT
# Copyright (c) 2026 TIANYANG LIU and BORAN CUI
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from event_features import field_features
ID = "field_year_id"
INDICES = ('ndvi', 'lswi', 'ndti')
EVENT = ['ndvi_pre', 'ndvi_pre_fraction_of_peak', 'ndvi_drydown_per_day', 'ndvi_step', 'lswi_pre', 'lswi_step', 'ndti_post', 'ndti_step', 'peak_to_harvest_days']
PEAK_OFFSETS = (0, 15, 30, 45, 60, 75)
RELATIVE = ['ndvi_plus_0d', 'lswi_plus_0d', 'ndti_plus_0d', 'ndvi_plus_15d', 'lswi_plus_15d', 'ndti_plus_15d', 'ndvi_plus_30d', 'lswi_plus_30d', 'ndti_plus_30d', 'ndvi_plus_45d', 'lswi_plus_45d', 'ndti_plus_45d', 'ndvi_plus_60d', 'lswi_plus_60d', 'ndti_plus_60d', 'ndvi_plus_75d', 'lswi_plus_75d', 'ndti_plus_75d']

def optical_features(observations):
    observations = observations.dropna(subset=list(INDICES)).sort_values("date")
    season = observations.loc[observations.date.dt.month.between(5, 11)]
    midsummer = season.loc[season.date.dt.month.between(6, 8)]
    late = season.loc[season.date.dt.month.between(9, 11)]
    features = dict.fromkeys(EVENT + RELATIVE, np.nan)
    features.update(optical_supported=False, valid_dates=len(season), peak_date=pd.NaT,
                    change_start=pd.NaT, change_end=pd.NaT)
    if len(season) < 5 or len(midsummer) < 2 or len(late) < 2:
        return features
    smooth = midsummer.ndvi.rolling(3, center=True, min_periods=2).median()
    if smooth.max() < .4:
        return features
    peak = midsummer.loc[smooth >= .95 * smooth.max(), "date"].iloc[-1]
    relative_day = (season.date - peak).dt.days
    for day in PEAK_OFFSETS:
        window = season.loc[abs(relative_day - day) <= 9]
        for index in INDICES:
            features[f"{index}_plus_{day}d"] = window[index].median() if len(window) else np.nan
    event = field_features(season)
    features.update({key: event[key] for key in EVENT})
    features.update(peak_date=peak, change_start=event["harvest_start"], change_end=event["harvest_end"],
                    optical_supported=True)
    return features

def classes(rows):
    return rows.label.eq("silage").to_numpy(int)

def fit_model(fields, covariates):
    branches = {}
    for name, base in (("event", EVENT), ("peak_aligned", RELATIVE)):
        supported = fields[base].notna().any(axis=1)
        train = fields.loc[supported]
        y = classes(train)
        if np.unique(y).size != 2:
            continue
        columns = [*base, *covariates]
        cell_count = train.groupby(["region", "label"])[ID].transform("size")
        weight = 1.0 / cell_count
        weight *= len(weight) / weight.sum()
        model = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(),
                              LogisticRegression(C=.3, max_iter=1000))
        model.fit(train[columns], y, standardscaler__sample_weight=weight,
                  logisticregression__sample_weight=weight)
        branches[name] = {"features": columns, "support": base, "estimator": model,
                          "training_examples": len(train)}
    if not branches:
        raise ValueError("No branch with both classes in training")
    return branches

def predict(fields, branches):
    probability = np.full(len(fields), np.nan)
    branch_name = np.full(len(fields), "uncertain", dtype=object)
    for name, branch in branches.items():
        valid = (fields.optical_supported.to_numpy() & np.isnan(probability)
                 & fields[branch["support"]].notna().any(axis=1).to_numpy())
        if valid.any():
            probability[valid] = branch["estimator"].predict_proba(
                fields.loc[valid, branch["features"]])[:, 1]
            branch_name[valid] = name
    return probability, branch_name
