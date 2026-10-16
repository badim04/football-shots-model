"""
Drift detection via Population Stability Index (PSI).

The baseline is the first `DRIFT_BASELINE_WEEKS` of the season — the
period the model's initial assumptions were formed on. Every retrain
after that compares the CURRENT week's feature distributions against that
same fixed baseline, not against "last week," because week-over-week
comparison drifts with the model and never catches slow, cumulative shift
(e.g. a league trending more defensive all season). PSI is the standard
industry metric for this because it's interpretable at fixed thresholds
regardless of the feature's scale.

PSI thresholds (standard convention):
  < 0.10  -> no meaningful shift
  0.10-0.25 -> moderate shift, worth a look
  > 0.25  -> significant shift, treat predictions from this feature with caution
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone

import config

DRIFT_FEATURES = [
    "overall_avg_shots", "overall_avg_sot",
    "opp_def_shots_conceded_avg", "opp_def_sot_conceded_avg",
]


def _psi(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    baseline = baseline.dropna()
    current = current.dropna()
    if len(baseline) < 10 or len(current) < 10:
        return 0.0

    edges = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    base_counts, _ = np.histogram(baseline, bins=edges)
    curr_counts, _ = np.histogram(current, bins=edges)

    base_pct = np.clip(base_counts / max(base_counts.sum(), 1), 1e-4, None)
    curr_pct = np.clip(curr_counts / max(curr_counts.sum(), 1), 1e-4, None)

    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def _flag(psi_value: float) -> str:
    if psi_value >= config.PSI_ALERT:
        return "alert"
    if psi_value >= config.PSI_WARNING:
        return "warning"
    return "ok"


def run_drift_check(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare the most recent week of feature rows against the season's
    baseline window. Returns a DataFrame ready for db.save_drift_metrics.
    """
    if feat_df.empty:
        return pd.DataFrame()

    dates = pd.to_datetime(feat_df["date"])
    baseline_cutoff = dates.min() + pd.Timedelta(weeks=config.DRIFT_BASELINE_WEEKS)
    current_cutoff = dates.max() - pd.Timedelta(weeks=1)

    baseline_df = feat_df[dates <= baseline_cutoff]
    current_df = feat_df[dates > current_cutoff]

    retrain_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for feature in DRIFT_FEATURES:
        if feature not in feat_df.columns:
            continue
        # If we're still inside the baseline window itself, there's nothing
        # to compare yet — report 0 rather than a misleading comparison.
        if current_df.empty or baseline_df.empty or baseline_cutoff >= dates.max():
            psi_value = 0.0
        else:
            psi_value = _psi(baseline_df[feature], current_df[feature])
        rows.append({
            "retrain_ts": retrain_ts,
            "feature_name": feature,
            "psi_score": round(psi_value, 4),
            "flag": _flag(psi_value),
        })

    return pd.DataFrame(rows)
