"""
Central configuration. Every threshold/window the app exposes as an
adjustable control reads its DEFAULT from here — the Streamlit app
overrides these at runtime, this file only sets what a fresh run starts with.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.environ.get("SHOTS_DB_PATH", os.path.join(DATA_DIR, "matches.db"))

# Raw FotMob match JSON, gzipped (~27 KB per match). This is what makes a
# re-parse free: if we later want a stat we didn't extract, it's already on
# disk and `scripts/backfill.py --reparse` rebuilds the tables with no network.
RAW_CACHE_DIR = os.path.join(DATA_DIR, "raw_cache")

for d in (DATA_DIR, MODELS_DIR, LOGS_DIR, RAW_CACHE_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
# The season predictions are made for. Change once per season.
CURRENT_SEASON = "2026/2027"

# Seasons the backfill pulls. Past seasons are what head-to-head history and
# long-run player baselines come from; a single season has at most one or two
# meetings between any two teams, which is too thin for H2H to mean anything.
# Each past season is roughly 2,300 match requests (~2 hours at the delays below).
BACKFILL_SEASONS = ["2023/2024", "2024/2025", "2025/2026", CURRENT_SEASON]

# ---------------------------------------------------------------------------
# Competitions in scope: FotMob league IDs (verified 2026-09-25 against
# fotmob.com/api/data/leagues?id=<ID>). European competitions are the main
# stage only — FotMob keeps qualifying rounds under separate IDs.
# ---------------------------------------------------------------------------
COMPETITIONS = {
    "Premier League": 47,
    "La Liga": 87,
    "Bundesliga": 54,
    "Serie A": 55,
    "Ligue 1": 53,
    "Champions League": 42,
    "Europa League": 73,
    "Conference League": 10216,
}

# ---------------------------------------------------------------------------
# Scraper politeness settings
# ---------------------------------------------------------------------------
MIN_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 4.0
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 30   # first retry waits ~30s, then doubles
MAX_DETAIL_ATTEMPTS = 3     # a match that fails this many runs in a row is left alone
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Feature engineering defaults (all overridable from the Streamlit sidebar)
# ---------------------------------------------------------------------------
RECENT_N_OVERALL = 5          # rolling window for a player's general form
RECENT_N_OPPONENT = 3         # rolling window for form vs this exact opponent
MIN_MINUTES_TO_COUNT = 45     # ignore appearances shorter than this (applied in features, not at storage)
MIN_SAMPLE_SIZE_WARNING = 3   # flag predictions built on fewer than N games

# ---------------------------------------------------------------------------
# Prediction thresholds (defaults — adjustable live in the app)
# ---------------------------------------------------------------------------
DEFAULT_SHOT_THRESHOLDS = [1, 2, 3]
DEFAULT_SOT_THRESHOLDS = [1, 2]
DEFAULT_MIN_PERCENTAGE = 0.60   # confidence (predicted probability) required to flag "eligible"

# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
PSI_WARNING = 0.10
PSI_ALERT = 0.25
DRIFT_BASELINE_WEEKS = 4   # first N weeks of the current season = the baseline distribution

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_SHOTS_PATH = os.path.join(MODELS_DIR, "model_shots.joblib")
MODEL_SOT_PATH = os.path.join(MODELS_DIR, "model_sot.joblib")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_list.json")
TEST_HOLDOUT_WEEKS = 3   # most recent N weeks held out for time-based validation
