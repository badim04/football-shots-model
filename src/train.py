"""
Training and inference.

Why Poisson on top of XGBoost, instead of just thresholding a point
estimate: shots and shots-on-target are small non-negative counts, which
is exactly what the Poisson distribution models. XGBoost predicts each
player's expected value (lambda) for the upcoming match; from that single
number we get a principled P(shots >= k) for *any* k via the Poisson
survival function, instead of an arbitrary rule like "predicted >= 1.5
counts as eligible for the 2+ line." This is also why "eligibility" here
is a probability crossing MIN_PERCENTAGE, not a hard cutoff on the raw
prediction — a player predicted 1.4 shots isn't obviously "in" or "out"
for the 2+ market, but P(shots >= 2 | lambda=1.4) is a real, comparable number.
"""

import json
import warnings
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

import config
from src.features import FEATURE_COLUMNS


def _time_based_split(df: pd.DataFrame, holdout_weeks: int):
    df = df.sort_values("date")
    max_date = pd.to_datetime(df["date"]).max()
    cutoff = max_date - timedelta(weeks=holdout_weeks)
    train_df = df[pd.to_datetime(df["date"]) <= cutoff]
    test_df = df[pd.to_datetime(df["date"]) > cutoff]
    # Guard against a too-early season where the holdout window would empty the train set.
    if len(train_df) < 30 or len(test_df) == 0:
        split_idx = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    return train_df, test_df


def train_and_save(feat_df: pd.DataFrame, holdout_weeks: int = None) -> dict:
    holdout_weeks = holdout_weeks or config.TEST_HOLDOUT_WEEKS
    train_df, test_df = _time_based_split(feat_df, holdout_weeks)

    X_train, X_test = train_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS]

    model_shots = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        objective="count:poisson", subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model_sot = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        objective="count:poisson", subsample=0.8, colsample_bytree=0.8, random_state=42,
    )

    model_shots.fit(X_train, train_df["target_shots"])
    model_sot.fit(X_train, train_df["target_sot"])

    mae_shots = mae_sot = None
    if len(test_df) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mae_shots = mean_absolute_error(test_df["target_shots"], model_shots.predict(X_test))
            mae_sot = mean_absolute_error(test_df["target_sot"], model_sot.predict(X_test))

    joblib.dump(model_shots, config.MODEL_SHOTS_PATH)
    joblib.dump(model_sot, config.MODEL_SOT_PATH)
    with open(config.FEATURE_LIST_PATH, "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    return {
        "retrain_ts": datetime.now(timezone.utc).isoformat(),
        "mae_shots": float(mae_shots) if mae_shots is not None else None,
        "mae_sot": float(mae_sot) if mae_sot is not None else None,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "holdout_from": str(test_df["date"].min()) if len(test_df) else None,
        "holdout_to": str(test_df["date"].max()) if len(test_df) else None,
    }


def load_models():
    model_shots = joblib.load(config.MODEL_SHOTS_PATH)
    model_sot = joblib.load(config.MODEL_SOT_PATH)
    return model_shots, model_sot


def predict(rows: pd.DataFrame, shot_thresholds: list, sot_thresholds: list,
            min_percentage: float) -> pd.DataFrame:
    """
    Score upcoming-match feature rows. Adds pred_shots, pred_sot (expected
    values), a probability for each requested threshold, and an
    `eligible_*` boolean column per threshold once that probability
    clears min_percentage.
    """
    model_shots, model_sot = load_models()
    X = rows[FEATURE_COLUMNS]

    rows = rows.copy()
    rows["pred_shots"] = model_shots.predict(X)
    rows["pred_sot"] = model_sot.predict(X)
    rows["pred_shots"] = rows["pred_shots"].clip(lower=0.01)
    rows["pred_sot"] = rows["pred_sot"].clip(lower=0.01)

    prob_records = []
    for _, r in rows.iterrows():
        probs = {}
        for k in shot_thresholds:
            probs[f"shots>={k}"] = float(1 - poisson.cdf(k - 1, r["pred_shots"]))
        for k in sot_thresholds:
            probs[f"sot>={k}"] = float(1 - poisson.cdf(k - 1, r["pred_sot"]))
        prob_records.append(probs)

    rows["prob_json"] = [json.dumps(p) for p in prob_records]
    for k in shot_thresholds:
        rows[f"eligible_shots_{k}+"] = [p[f"shots>={k}"] >= min_percentage for p in prob_records]
    for k in sot_thresholds:
        rows[f"eligible_sot_{k}+"] = [p[f"sot>={k}"] >= min_percentage for p in prob_records]

    keep_cols = [
        "match_id", "date", "competition", "player_id", "player_name", "team_id", "team", "opponent",
        "pred_shots", "pred_sot", "overall_sample_size", "vs_opp_sample_size", "prob_json",
    ] + [c for c in rows.columns if c.startswith("eligible_")]
    out = rows[keep_cols].rename(columns={
        "overall_sample_size": "sample_size_overall", "vs_opp_sample_size": "sample_size_vs_opp",
    })
    return out.sort_values("pred_shots", ascending=False)
