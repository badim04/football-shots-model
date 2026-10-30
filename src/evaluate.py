"""
Track record: how the predictions that were actually published did.

For every (match, player), the last prediction made BEFORE kickoff is
compared with what the player did. This replaces the old PSI drift check
as the health signal: drift only says the inputs changed; calibration says
whether "70%" picks really land about 70% of the time, which is the thing
that matters for these predictions.
"""

import json

import numpy as np
import pandas as pd

import config
from src import db


def scored_predictions(since: str = None) -> pd.DataFrame:
    """Pre-kickoff predictions joined to actual outcomes (completed matches only)."""
    preds = db.latest_predictions_before_kickoff()
    if preds.empty:
        return pd.DataFrame()
    if since:
        preds = preds[preds["date"] >= since]
    done = db.load_matches(status="completed")
    done = done[done.details_status == "ok"][["match_id"]]
    preds = preds.merge(done, on="match_id")
    if preds.empty:
        return pd.DataFrame()
    actual = db.load_player_match_stats()[["match_id", "player_id", "minutes", "shots", "shots_on_target"]]
    out = preds.merge(actual, on=["match_id", "player_id"], how="left")
    out["played"] = out["minutes"].fillna(0) > 0
    return out


def track_record(min_prob: float = None, since: str = None) -> pd.DataFrame:
    """
    Per betting line: how many "picks" (probability ≥ min_prob) there were,
    how many landed, and the average predicted probability. Players who
    didn't play are excluded (they're reported separately by `coverage`).
    """
    min_prob = config.DEFAULT_MIN_PERCENTAGE if min_prob is None else min_prob
    df = scored_predictions(since)
    if df.empty:
        return pd.DataFrame()
    df = df[df.played]
    rows = []
    probs = df["prob_json"].apply(json.loads)
    for line in sorted({k for p in probs for k in p}):
        target, k = line.split(">=")
        actual_col = "shots" if target == "shots" else "shots_on_target"
        p = probs.apply(lambda d: d.get(line, np.nan))
        picks = p >= min_prob
        if not picks.any():
            continue
        hits = df.loc[picks, actual_col] >= int(k)
        rows.append({"line": line, "picks": int(picks.sum()), "hits": int(hits.sum()),
                     "hit_rate": float(hits.mean()), "avg_prob": float(p[picks].mean())})
    return pd.DataFrame(rows)


def calibration(line: str = "shots>=1", since: str = None,
                bins=(0, .2, .4, .5, .6, .7, .8, 1.0)) -> pd.DataFrame:
    """Predicted vs actual hit rate across ALL predictions for one line."""
    df = scored_predictions(since)
    if df.empty:
        return pd.DataFrame()
    df = df[df.played]
    target, k = line.split(">=")
    actual_col = "shots" if target == "shots" else "shots_on_target"
    p = df["prob_json"].apply(lambda s: json.loads(s).get(line, np.nan))
    t = pd.DataFrame({"p": p, "hit": (df[actual_col] >= int(k)).astype(float)}).dropna()
    t["bucket"] = pd.cut(t["p"], bins=list(bins), include_lowest=True).astype(str)
    return (t.groupby("bucket", observed=True)
            .agg(n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean")).reset_index())


def coverage(since: str = None) -> dict:
    """How often a predicted starter actually played (lineup accuracy)."""
    df = scored_predictions(since)
    if df.empty:
        return {}
    by_type = df.groupby(df["lineup_type"].fillna("estimated"))["played"].agg(["size", "mean"])
    return {k: {"predicted_players": int(v["size"]), "played_rate": float(v["mean"])}
            for k, v in by_type.iterrows()}
