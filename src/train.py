"""
Training, model selection and inference.

Three candidate models predict each starter's expected shots (and,
separately, shots on target), λ:

- baseline  Recent form blended with the player's normal rate, × his usual
            minutes, × how much the opponent concedes. No learning beyond
            one scale factor. Anything fancier has to beat this to be used.
- glm       Poisson regression (log link) on the features, several of them
            log-transformed so effects multiply, as they do for rates.
- xgb       Gradient-boosted trees with a Poisson objective — can pick up
            interactions (e.g. form matters more for forwards).

Each is fitted on everything except the most recent TEST_HOLDOUT_WEEKS and
scored on those weeks. The best (lowest Poisson deviance) is refitted on
all data and saved. Nothing is ever shuffled — validation is always
"train on the past, test on what came after".

Probabilities. P(shots ≥ k) comes from a count distribution around λ.
Shots are usually more spread out than Poisson allows (a player's 2.0
average is more often 0 or 5 than Poisson expects), so a negative binomial
is fitted on the holdout too and used when it describes the outcomes
better. This matters most for the 2+ and 3+ lines.
"""

import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import config
from src.features import FEATURE_COLUMNS

TARGETS = {"shots": "target_shots", "sot": "target_sot"}
EVAL_THRESHOLDS = {"shots": [1, 2, 3], "sot": [1, 2]}
MAX_EXPECTED = {"shots": 12.0, "sot": 6.0}
LOG_COLUMNS = [
    "form_shots_p90", "form_sot_p90", "form_xg_p90", "form_box_p90", "baseline_shots_p90",
    "baseline_sot_p90", "h2h_team_ratio", "h2h_player_ratio", "team_shots_for",
    "opp_shots_against_rel", "opp_shots_for", "league_avg_shots", "form_minutes",
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BaselineModel:
    def __init__(self, target: str):
        self.target = target
        self.scale = 1.0

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        col = "shots" if self.target == "shots" else "sot"
        n = X["form_games"]
        rate = (n * X[f"form_{col}_p90"] + 3 * X[f"baseline_{col}_p90"]) / (n + 3)
        return (rate * X["form_minutes"].clip(upper=90) / 90 * X["opp_shots_against_rel"]).values

    def fit(self, X, y):
        self.scale = float(np.sum(y) / max(np.sum(self._raw(X)), 1e-9))
        return self

    def predict(self, X):
        return self._raw(X) * self.scale


class GLMModel:
    def __init__(self, features):
        self.features = list(features)
        self.pipe = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=3000))

    def _matrix(self, X):
        M = X[self.features].astype(float).copy()
        for c in self.features:
            if c in LOG_COLUMNS:
                M[c] = np.log(M[c].clip(lower=0) + 0.05)
        return M.fillna(0).values

    def fit(self, X, y):
        M = self._matrix(X)
        # a log-linear model explodes on values it never saw (e.g. H2H ratios
        # trained when every one was 1.0), so predictions are clipped to the
        # training range
        self.lo, self.hi = np.nanpercentile(M, 0.5, axis=0), np.nanpercentile(M, 99.5, axis=0)
        self.pipe.fit(M, y)
        return self

    def predict(self, X):
        M = self._matrix(X)
        if hasattr(self, "lo"):
            M = np.clip(M, self.lo, self.hi)
        return self.pipe.predict(M)


class XGBModel:
    def __init__(self, features):
        self.features = list(features)
        self.model = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.04, min_child_weight=20,
            objective="count:poisson", subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )

    def fit(self, X, y):
        self.model.fit(X[self.features].astype(float), y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.features].astype(float))


def make_model(kind: str, target: str, features=FEATURE_COLUMNS):
    if kind == "baseline":
        return BaselineModel(target)
    if kind == "glm":
        return GLMModel(features)
    if kind == "xgb":
        return XGBModel(features)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def prob_at_least(lam, k: int, alpha: float = 0.0):
    """P(X >= k) for mean λ: Poisson when alpha == 0, else negative binomial."""
    lam = np.clip(np.asarray(lam, dtype=float), 1e-6, None)
    if alpha and alpha > 1e-6:
        n = 1.0 / alpha
        return nbinom.sf(k - 1, n, n / (n + lam))
    return poisson.sf(k - 1, lam)


def _loglik(y, lam, alpha):
    lam = np.clip(lam, 1e-6, None)
    if alpha <= 1e-6:
        return float(np.sum(poisson.logpmf(y, lam)))
    n = 1.0 / alpha
    return float(np.sum(nbinom.logpmf(y, n, n / (n + lam))))


def fit_dispersion(y, lam) -> float:
    """Negative-binomial alpha by maximum likelihood (0 = Poisson is at least as good)."""
    res = minimize_scalar(lambda a: -_loglik(y, lam, a), bounds=(1e-4, 3.0), method="bounded")
    return float(res.x) if -res.fun > _loglik(y, lam, 0.0) + 1.0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def time_split(rows: pd.DataFrame, holdout_weeks: int = None):
    holdout_weeks = holdout_weeks or config.TEST_HOLDOUT_WEEKS
    dates = pd.to_datetime(rows["date"])
    cutoff = dates.max() - timedelta(weeks=holdout_weeks)
    train_df, test_df = rows[dates <= cutoff], rows[dates > cutoff]
    if len(train_df) < 500 or test_df.empty:   # too early to hold out whole weeks
        idx = int(len(rows) * 0.8)
        ordered = rows.sort_values("date")
        train_df, test_df = ordered.iloc[:idx], ordered.iloc[idx:]
    return train_df, test_df


def score(y, lam, target: str, alpha: float) -> dict:
    """Deviance plus, per betting line, log-loss and Brier of P(X >= k)."""
    y, lam = np.asarray(y, dtype=float), np.clip(np.asarray(lam, dtype=float), 1e-6, None)
    out = {"deviance": float(mean_poisson_deviance(y, lam)),
           "mean_pred": float(lam.mean()), "mean_actual": float(y.mean())}
    for k in EVAL_THRESHOLDS[target]:
        p = np.clip(prob_at_least(lam, k, alpha), 1e-6, 1 - 1e-6)
        hit = (y >= k).astype(float)
        out[f"logloss_{k}+"] = float(-np.mean(hit * np.log(p) + (1 - hit) * np.log(1 - p)))
        out[f"brier_{k}+"] = float(np.mean((p - hit) ** 2))
    return out


def calibration_table(y, lam, k: int, alpha: float, bins=(0, .2, .4, .5, .6, .7, .8, 1.0)) -> pd.DataFrame:
    """Predicted vs actual hit rate for P(X >= k), bucketed by predicted probability."""
    p = prob_at_least(lam, k, alpha)
    df = pd.DataFrame({"p": p, "hit": (np.asarray(y) >= k).astype(float)})
    df["bucket"] = pd.cut(df["p"], bins=list(bins), include_lowest=True)
    return (df.groupby("bucket", observed=True)
            .agg(n=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean")).reset_index())


def compare_models(rows: pd.DataFrame, kinds=("baseline", "glm", "xgb"),
                   features=FEATURE_COLUMNS, holdout_weeks: int = None) -> dict:
    """Fit each kind on the past, score on the holdout weeks. Returns per-target results."""
    train_df, test_df = time_split(rows, holdout_weeks)
    results = {"n_train": len(train_df), "n_test": len(test_df),
               "holdout_from": str(test_df["date"].min()), "holdout_to": str(test_df["date"].max()),
               "targets": {}}
    for target, col in TARGETS.items():
        per_kind = {}
        for kind in kinds:
            model = make_model(kind, target, features).fit(train_df, train_df[col].values)
            lam = model.predict(test_df)
            alpha = fit_dispersion(test_df[col].values, lam)
            per_kind[kind] = {"alpha": alpha, **score(test_df[col].values, lam, target, alpha),
                              "_lam": lam}
        results["targets"][target] = per_kind
    results["_test"] = test_df
    return results


# ---------------------------------------------------------------------------
# Train + save, load, predict
# ---------------------------------------------------------------------------

def train_and_save(rows: pd.DataFrame) -> dict:
    """
    Pick the best model per target on the holdout, refit it on all rows,
    and save a bundle with everything inference needs.
    """
    comparison = compare_models(rows)
    test_df = comparison.pop("_test")
    bundle = {"trained_at": datetime.now(timezone.utc).isoformat(), "features": FEATURE_COLUMNS,
              "n_rows": len(rows), "data_to": str(rows["date"].max()), "targets": {}}
    report = {k: v for k, v in comparison.items() if k != "targets"}
    report["targets"] = {}

    for target, col in TARGETS.items():
        per_kind = comparison["targets"][target]
        best = min(per_kind, key=lambda k: per_kind[k]["deviance"])
        final = make_model(best, target).fit(rows, rows[col].values)
        bundle["targets"][target] = {"kind": best, "model": final, "alpha": per_kind[best]["alpha"]}
        report["targets"][target] = {
            "chosen": best,
            "models": {k: {m: v for m, v in r.items() if not m.startswith("_")} for k, r in per_kind.items()},
            "calibration": {f"{k}+": calibration_table(test_df[col].values, per_kind[best]["_lam"], k,
                                                        per_kind[best]["alpha"]).to_dict("records")
                            for k in EVAL_THRESHOLDS[target]},
        }

    bundle["report"] = report
    joblib.dump(bundle, config.MODEL_BUNDLE_PATH)
    return report


def load_bundle() -> dict:
    return joblib.load(config.MODEL_BUNDLE_PATH)


def predict(rows: pd.DataFrame, shot_thresholds: list, sot_thresholds: list,
            bundle: dict = None) -> pd.DataFrame:
    """Expected shots/SOT and P(≥k) for every requested line, per row."""
    bundle = bundle or load_bundle()
    rows = rows.copy()
    lams, probs = {}, {}
    for target, thresholds in (("shots", shot_thresholds), ("sot", sot_thresholds)):
        t = bundle["targets"][target]
        # last-resort cap: no outfield starter averages anywhere near 12 shots
        lams[target] = np.clip(t["model"].predict(rows), 0.01, MAX_EXPECTED[target])
        if target == "sot":
            lams["sot"] = np.minimum(lams["sot"], lams["shots"])
        for k in thresholds:
            probs[f"{target}>={k}"] = prob_at_least(lams[target], k, t["alpha"])
    rows["pred_shots"], rows["pred_sot"] = lams["shots"], lams["sot"]
    rows["prob_json"] = [json.dumps({key: round(float(v[i]), 4) for key, v in probs.items()})
                         for i in range(len(rows))]
    rows["model_kind"] = f"{bundle['targets']['shots']['kind']}/{bundle['targets']['sot']['kind']}"
    return rows
