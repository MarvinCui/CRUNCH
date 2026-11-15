# SPDX-License-Identifier: MIT
# Copyright (c) 2026 TIANYANG LIU and BORAN CUI
"""Summarize field observations around a candidate harvest event."""
import numpy as np
import pandas as pd


INDEXES = ("ndvi", "lswi", "ndre", "ndti", "gray", "gray_iqr")
DATE_FEATURES = ("harvest_doy", "peak_to_harvest_days")
CANOPY_FEATURES = (
    "ndvi_pre", "ndvi_pre_fraction_of_peak", "ndvi_drydown_per_day", "ndvi_step",
    "lswi_pre", "lswi_step", "ndre_pre", "ndre_step",
)
RESIDUE_FEATURES = ("ndti_post", "ndti_step")
GRAY_FEATURES = ("gray_step", "gray_iqr_step")
MODEL_FEATURES = DATE_FEATURES + CANOPY_FEATURES + RESIDUE_FEATURES + GRAY_FEATURES


def finite_median(values):
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def field_features(observations, *, use_hpi=False):
    """Rank sustained declines after the crop peak using actual image dates.

    The reported interval brackets the transition between observations. The
    midpoint is used for date-error comparisons only.
    """
    rows = observations.dropna(subset=["ndvi"]).sort_values("date")
    result = dict.fromkeys(MODEL_FEATURES, np.nan)
    result.update(harvest_status="insufficient_observations", observation_count=len(rows),
                  harvest_start=pd.NaT, harvest_end=pd.NaT, harvest_date=pd.NaT,
                  harvest_interval_days=np.nan, peak_ndvi=np.nan)
    if len(rows) < 7:
        return result
    days = rows.date.dt.dayofyear.to_numpy()
    ndvi = rows.ndvi.to_numpy(float)
    # Smooth over neighboring observations to limit isolated spikes.
    smooth = pd.Series(ndvi).rolling(3, center=True, min_periods=2).median().to_numpy()
    peak = int(np.argmax(smooth))
    result.update(peak_ndvi=float(smooth[peak]), harvest_status="no_event")
    if smooth[peak] < 0.4:
        return result

    candidates = []
    hpi = rows.hpi.to_numpy(float) if use_hpi and "hpi" in rows else None
    for i in range(peak, len(rows) - 2):
        gap = days[i + 1] - days[i]
        if not 0 < gap <= 18 or days[i + 2] - days[i + 1] > 24:
            continue
        ndvi_drop = ndvi[i] - ndvi[i + 1]
        persistence = ndvi[i] - max(ndvi[i + 1:i + 3])
        if ndvi_drop >= 0.09 and persistence >= 0.07:
            score = (persistence + ndvi_drop) / np.sqrt(gap)
            candidates.append((score, i, "ndvi_step"))
        if hpi is None or not np.isfinite(hpi[i:i + 3]).all():
            continue
        # Experimental, unnormalized NIR/NDVI corroboration, with a check for
        # the ratio's instability as NDVI approaches zero.
        hpi_rise = min(hpi[i + 1:i + 3]) - hpi[i]
        if ndvi[i + 1] >= .16 and ndvi_drop >= -.025 and hpi_rise >= .2 and hpi[i] <= 1.15:
            score = (max(.0, persistence) + hpi_rise * .55) / np.sqrt(gap)
            candidates.append((score, i, "hpi_rise"))
    if not candidates:
        return result

    _, i, detector = max(candidates)
    start, end = rows.date.iloc[i:i + 2]
    midpoint = start + (end - start) / 2
    prior = (days >= days[i] - 14) & (days <= days[i])
    later = (days >= days[i + 1]) & (days <= days[i + 1] + 12)
    drydown = (days >= days[i] - 28) & (days <= days[i])
    if drydown.sum() >= 3 and np.ptp(days[drydown]) >= 8:
        result["ndvi_drydown_per_day"] = float(np.polyfit(days[drydown], ndvi[drydown], 1)[0])
    for key in INDEXES:
        if key not in rows:
            continue
        values = rows[key].to_numpy(float)
        pre = finite_median(values[prior])
        post = finite_median(values[later])
        result[f"{key}_pre"] = pre
        result[f"{key}_post"] = post
        result[f"{key}_step"] = post - pre
    # A dry surface just after collection is the cleanest residue comparison.
    # Use the first observed post-event NDTI and retain its NDVI for context.
    if "ndti" in rows:
        result["ndti_post"] = float(rows.ndti.iloc[i + 1])
        result["ndti_step"] = result["ndti_post"] - result["ndti_pre"]
    result.update(harvest_status="detected", harvest_detector=detector, harvest_start=start, harvest_end=end,
                  harvest_date=midpoint, harvest_doy=(days[i] + days[i + 1]) / 2,
                  harvest_interval_days=days[i + 1] - days[i],
                  peak_to_harvest_days=(days[i] + days[i + 1]) / 2 - days[peak],
                  ndvi_pre_fraction_of_peak=result["ndvi_pre"] / smooth[peak])
    return result


def extract_field_features(observations, *, use_hpi=False):
    observations = observations.copy()
    observations["date"] = pd.to_datetime(observations.date)
    if observations.duplicated(["field_id", "year", "date"]).any():
        raise ValueError("Expected one observation per field, year and date")
    if not (observations.date.dt.year == observations.year).all():
        raise ValueError("Observation dates must match their field year")
    return pd.DataFrame([
        {"field_id": field, "year": year, **field_features(rows, use_hpi=use_hpi)}
        for (field, year), rows in observations.groupby(["field_id", "year"], sort=False)
    ])
