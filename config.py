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
# Feature engineering
# ---------------------------------------------------------------------------
# Recent form is an exponentially weighted average over the player's prior
# appearances: with a half-life of 3 games, last game counts 1.0, three games
# ago 0.5, six games ago 0.25 — so the last 4-7 games carry nearly all the
# weight. Per 90 minutes, so being subbed at 60' isn't mistaken for bad form.
FORM_HALFLIFE_GAMES = 3
LONG_RUN_HALFLIFE_GAMES = 20     # the slower "what is this player normally" rate
TEAM_FORM_HALFLIFE_GAMES = 4     # team attack/defence form
H2H_HALFLIFE_MEETINGS = 2        # most recent meetings count most
H2H_SHRINK_MEETINGS = 2          # H2H effect is pulled toward "no effect" until there are a few meetings
BASELINE_PRIOR_MINUTES = 450     # a player with little history is pulled toward his position's average
MIN_SAMPLE_SIZE_WARNING = 3      # flag predictions built on fewer than N prior games

# Derbies, by FotMob team name. H2H history already captures how a pairing
# tends to go; this flag adds the "it's a derby" effect for pairings with
# few meetings in the data.
DERBIES = [
    ("Arsenal", "Tottenham Hotspur"), ("Arsenal", "Chelsea"), ("Chelsea", "Tottenham Hotspur"),
    ("Liverpool", "Everton"), ("Liverpool", "Manchester United"),
    ("Manchester City", "Manchester United"), ("Newcastle United", "Sunderland"),
    ("Crystal Palace", "Brighton & Hove Albion"), ("Aston Villa", "Birmingham City"),
    ("Real Madrid", "Atlético Madrid"), ("Real Madrid", "Barcelona"), ("Barcelona", "Espanyol"),
    ("Sevilla", "Real Betis"), ("Athletic Club", "Real Sociedad"), ("Valencia", "Villarreal"),
    ("Celta Vigo", "Deportivo A Coruña"),
    ("Milan", "Inter"), ("Inter", "Juventus"), ("Roma", "Lazio"), ("Juventus", "Torino"),
    ("Genoa", "Sampdoria"), ("Napoli", "Roma"),
    ("Bayern München", "Borussia Dortmund"), ("Borussia Dortmund", "Schalke 04"),
    ("Hamburger SV", "Werder Bremen"), ("1. FC Köln", "Borussia Mönchengladbach"),
    ("Paris Saint-Germain", "Marseille"), ("Lyon", "Saint-Étienne"), ("Nice", "Monaco"),
    ("Lens", "Lille"), ("Paris Saint-Germain", "Paris FC"),
]

# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------
# Small files the deployed app reads (the full database stays local).
PUBLISHED_DIR = os.path.join(DATA_DIR, "published")
os.makedirs(PUBLISHED_DIR, exist_ok=True)

# Telegram: the bot token and chat id are read from the environment or from
# a local, git-ignored `.env` file — never from this file.
TELEGRAM_MAX_PICKS_PER_MATCH = 4
TELEGRAM_FOOTER = ("Model probabilities from recent form, head-to-head and team context. "
                   "Lineups are FotMob's predicted XI unless marked confirmed. Not betting advice.")

# ---------------------------------------------------------------------------
# Prediction thresholds (defaults — adjustable live in the app)
# ---------------------------------------------------------------------------
DEFAULT_SHOT_THRESHOLDS = [1, 2, 3]
DEFAULT_SOT_THRESHOLDS = [1, 2]
DEFAULT_MIN_PERCENTAGE = 0.60   # confidence (predicted probability) required to flag "eligible"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_BUNDLE_PATH = os.path.join(MODELS_DIR, "model_bundle.joblib")
TEST_HOLDOUT_WEEKS = 4   # most recent N weeks held out for model selection / evaluation
