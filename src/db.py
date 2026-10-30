"""
All persistence goes through this module. SQLite is enough at this scale
(a few seasons, ~8 competitions, weekly appends) and keeps the whole
project file-based.

Players and teams are keyed by FotMob's numeric IDs, not names — two
different players called "Danilo", or one club spelled two ways across
competitions, would otherwise silently merge into one history.
"""

import contextlib
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

import config

SCHEMA_VERSION = 2   # 1 = the original FBref schema (name-keyed)

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id            TEXT PRIMARY KEY,
    season              TEXT NOT NULL,
    competition         TEXT NOT NULL,
    league_id           INTEGER NOT NULL,
    round               TEXT,
    date                TEXT NOT NULL,          -- UTC date of kickoff, YYYY-MM-DD
    kickoff_utc         TEXT,
    home_team_id        INTEGER NOT NULL,
    home_team           TEXT NOT NULL,
    away_team_id        INTEGER NOT NULL,
    away_team           TEXT NOT NULL,
    home_score          INTEGER,
    away_score          INTEGER,
    status              TEXT NOT NULL,          -- scheduled | live | completed | cancelled | awarded
    fixtures_synced_at  TEXT,
    -- Per-match detail scrape state. A match only counts as done once its
    -- player stats are actually stored ('ok'), so a failed or empty parse
    -- is retried on the next run instead of being forgotten.
    details_status      TEXT NOT NULL DEFAULT 'pending',   -- pending | ok | no_stats | failed
    details_attempts    INTEGER NOT NULL DEFAULT 0,
    details_error       TEXT,
    details_fetched_at  TEXT
);
CREATE INDEX IF NOT EXISTS ix_matches_date ON matches(date);

CREATE TABLE IF NOT EXISTS player_match_stats (
    match_id         TEXT NOT NULL,
    player_id        INTEGER NOT NULL,
    player_name      TEXT NOT NULL,
    team_id          INTEGER NOT NULL,
    team             TEXT NOT NULL,
    opponent_id      INTEGER NOT NULL,
    opponent         TEXT NOT NULL,
    is_home          INTEGER NOT NULL,
    date             TEXT NOT NULL,
    competition      TEXT NOT NULL,
    season           TEXT NOT NULL,
    started          INTEGER,                   -- NULL when FotMob had no lineup
    position         TEXT,                      -- GK | DEF | MID | FWD (usual position)
    position_id      INTEGER,                   -- FotMob's slot for this match
    minutes          INTEGER NOT NULL,
    shots            INTEGER NOT NULL,
    shots_on_target  INTEGER NOT NULL,
    penalty_shots    INTEGER NOT NULL,
    goals            INTEGER,
    xg               REAL,
    npxg             REAL,
    touches_opp_box  REAL,
    rating           REAL,
    PRIMARY KEY (match_id, player_id)
);
CREATE INDEX IF NOT EXISTS ix_pms_player ON player_match_stats(player_id, date);

CREATE TABLE IF NOT EXISTS team_match_stats (
    match_id         TEXT NOT NULL,
    team_id          INTEGER NOT NULL,
    team             TEXT NOT NULL,
    opponent_id      INTEGER NOT NULL,
    opponent         TEXT NOT NULL,
    is_home          INTEGER NOT NULL,
    date             TEXT NOT NULL,
    competition      TEXT NOT NULL,
    season           TEXT NOT NULL,
    goals_for        INTEGER,
    goals_against    INTEGER,
    shots_for        REAL,
    shots_against    REAL,
    sot_for          REAL,
    sot_against      REAL,
    xg_for           REAL,
    xg_against       REAL,
    possession       REAL,
    touches_opp_box  REAL,
    big_chances      REAL,
    formation        TEXT,
    coach_id         INTEGER,
    coach_name       TEXT,
    PRIMARY KEY (match_id, team_id)
);

CREATE TABLE IF NOT EXISTS player_unavailability (
    match_id         TEXT NOT NULL,
    team_id          INTEGER NOT NULL,
    player_id        INTEGER NOT NULL,
    player_name      TEXT,
    date             TEXT NOT NULL,
    competition      TEXT NOT NULL,
    season           TEXT NOT NULL,
    reason           TEXT,                      -- injury | suspension | ...
    expected_return  TEXT,
    PRIMARY KEY (match_id, player_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    generated_at     TEXT NOT NULL,
    match_id         TEXT NOT NULL,
    player_id        INTEGER NOT NULL,
    player_name      TEXT NOT NULL,
    team_id          INTEGER NOT NULL,
    team             TEXT NOT NULL,
    opponent         TEXT NOT NULL,
    date             TEXT NOT NULL,
    competition      TEXT NOT NULL,
    pred_shots       REAL,
    pred_sot         REAL,
    sample_size_overall  INTEGER,
    sample_size_vs_opp   INTEGER,
    prob_json        TEXT,     -- JSON: {"shots>=1": 0.82, "sot>=2": 0.31, ...}
    PRIMARY KEY (generated_at, match_id, player_id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    trained_at       TEXT PRIMARY KEY,
    report_json      TEXT      -- holdout comparison, chosen model, calibration
);

CREATE TABLE IF NOT EXISTS telegram_posts (
    match_id         TEXT NOT NULL,
    kind             TEXT NOT NULL,     -- 'picks'
    lineup_type      TEXT,              -- what the post was based on: predicted | confirmed
    posted_at        TEXT NOT NULL,
    message_ids      TEXT,
    PRIMARY KEY (match_id, kind)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    retrain_ts       TEXT PRIMARY KEY,
    mae_shots        REAL,
    mae_sot          REAL,
    n_train_rows     INTEGER,
    n_test_rows      INTEGER,
    holdout_from     TEXT,
    holdout_to       TEXT
);

CREATE TABLE IF NOT EXISTS drift_metrics (
    retrain_ts       TEXT NOT NULL,
    feature_name     TEXT NOT NULL,
    psi_score        REAL,
    flag             TEXT,     -- 'ok' | 'warning' | 'alert'
    PRIMARY KEY (retrain_ts, feature_name)
);
"""

MATCH_FIXTURE_COLS = [
    "match_id", "season", "competition", "league_id", "round", "date", "kickoff_utc",
    "home_team_id", "home_team", "away_team_id", "away_team",
    "home_score", "away_score", "status", "fixtures_synced_at",
]
PLAYER_COLS = [
    "match_id", "player_id", "player_name", "team_id", "team", "opponent_id", "opponent",
    "is_home", "date", "competition", "season", "started", "position", "position_id",
    "minutes", "shots", "shots_on_target", "penalty_shots", "goals", "xg", "npxg",
    "touches_opp_box", "rating",
]
TEAM_COLS = [
    "match_id", "team_id", "team", "opponent_id", "opponent", "is_home", "date",
    "competition", "season", "goals_for", "goals_against", "shots_for", "shots_against",
    "sot_for", "sot_against", "xg_for", "xg_against", "possession", "touches_opp_box",
    "big_chances", "formation", "coach_id", "coach_name",
]
UNAVAILABLE_COLS = [
    "match_id", "team_id", "player_id", "player_name", "date", "competition", "season",
    "reason", "expected_return",
]
PREDICTION_COLS = [
    "generated_at", "match_id", "player_id", "player_name", "team_id", "team", "opponent",
    "date", "competition", "pred_shots", "pred_sot", "sample_size_overall",
    "sample_size_vs_opp", "prob_json", "kickoff_utc", "position", "lineup_type", "model_kind",
]

# Columns added after a table was first created: (table, column, type).
# init_db adds any that are missing, so existing databases keep their data.
ADDED_COLUMNS = [
    ("predictions", "kickoff_utc", "TEXT"),
    ("predictions", "position", "TEXT"),
    ("predictions", "lineup_type", "TEXT"),
    ("predictions", "model_kind", "TEXT"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def get_conn():
    # A generous timeout: the backfill may be writing while the app or a
    # prediction run reads/writes.
    conn = sqlite3.connect(config.DB_PATH, timeout=60)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create every table if it doesn't exist yet. Safe to call every run."""
    with get_conn() as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        has_tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='player_match_stats'"
        ).fetchone()[0]
        if has_tables and version < SCHEMA_VERSION:
            raise RuntimeError(
                f"{config.DB_PATH} uses an older schema (v{version}, name-keyed FBref data). "
                "Move or delete it and re-run the backfill (scripts/backfill.py)."
            )
        conn.executescript(SCHEMA)
        for table, column, col_type in ADDED_COLUMNS:
            existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _rows(df: pd.DataFrame, cols: list):
    """DataFrame rows as plain tuples with NaN/NA turned into NULL."""
    sub = df.reindex(columns=cols).astype(object)
    return list(sub.where(sub.notna(), None).itertuples(index=False, name=None))


def _insert(conn, table: str, df: pd.DataFrame, cols: list):
    if df is None or df.empty:
        return
    placeholders = ", ".join("?" for _ in cols)
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
        _rows(df, cols),
    )


# ---------------------------------------------------------------------------
# Matches (fixture lists + detail-scrape state)
# ---------------------------------------------------------------------------

def upsert_matches(df: pd.DataFrame):
    """
    Insert new fixtures / update known ones from a fixture-list sync.
    Only fixture columns are touched — the details_* scrape state of a
    match survives a re-sync.
    """
    if df.empty:
        return
    updates = ", ".join(f"{c}=excluded.{c}" for c in MATCH_FIXTURE_COLS if c != "match_id")
    with get_conn() as conn:
        conn.executemany(
            f"INSERT INTO matches ({', '.join(MATCH_FIXTURE_COLS)}) "
            f"VALUES ({', '.join('?' for _ in MATCH_FIXTURE_COLS)}) "
            f"ON CONFLICT(match_id) DO UPDATE SET {updates}",
            _rows(df, MATCH_FIXTURE_COLS),
        )


def matches_needing_details(seasons: list = None, competitions: list = None) -> pd.DataFrame:
    """
    Completed matches whose player stats aren't stored yet, oldest first.
    'no_stats' is retried too (FotMob can publish stats a while after the
    final whistle) until MAX_DETAIL_ATTEMPTS, after which it's accepted as
    a match FotMob simply has no player data for.
    """
    q = ("SELECT * FROM matches WHERE status = 'completed' "
         "AND details_status IN ('pending', 'failed', 'no_stats') AND details_attempts < ?")
    params = [config.MAX_DETAIL_ATTEMPTS]
    if seasons:
        q += f" AND season IN ({', '.join('?' for _ in seasons)})"
        params += list(seasons)
    if competitions:
        q += f" AND competition IN ({', '.join('?' for _ in competitions)})"
        params += list(competitions)
    q += " ORDER BY date, match_id"
    with get_conn() as conn:
        return pd.read_sql(q, conn, params=params)


def save_match_details(match_id: str, parsed: dict):
    """
    Store one match's parsed player/team/unavailability rows and mark it
    done, all in one transaction — a crash mid-write leaves the match
    'pending', never half-stored.
    """
    status = "ok" if not parsed["players"].empty else "no_stats"
    with get_conn() as conn:
        for table in ("player_match_stats", "team_match_stats", "player_unavailability"):
            conn.execute(f"DELETE FROM {table} WHERE match_id = ?", (match_id,))
        _insert(conn, "player_match_stats", parsed["players"], PLAYER_COLS)
        _insert(conn, "team_match_stats", parsed["teams"], TEAM_COLS)
        _insert(conn, "player_unavailability", parsed["unavailable"], UNAVAILABLE_COLS)
        conn.execute(
            "UPDATE matches SET details_status = ?, details_error = NULL, details_fetched_at = ?, "
            "details_attempts = details_attempts + 1 WHERE match_id = ?",
            (status, _now(), match_id),
        )
    return status


def mark_details_failed(match_id: str, error: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE matches SET details_status = 'failed', details_error = ?, "
            "details_attempts = details_attempts + 1 WHERE match_id = ?",
            (error[:500], match_id),
        )


def reset_detail_attempts(seasons: list = None):
    """Give previously given-up matches another go (e.g. after a parser fix)."""
    q = "UPDATE matches SET details_attempts = 0 WHERE details_status IN ('failed', 'no_stats')"
    params = []
    if seasons:
        q += f" AND season IN ({', '.join('?' for _ in seasons)})"
        params = list(seasons)
    with get_conn() as conn:
        conn.execute(q, params)


def load_matches(status: str = None, seasons: list = None) -> pd.DataFrame:
    q, params = "SELECT * FROM matches WHERE 1=1", []
    if status:
        q += " AND status = ?"
        params.append(status)
    if seasons:
        q += f" AND season IN ({', '.join('?' for _ in seasons)})"
        params += list(seasons)
    with get_conn() as conn:
        return pd.read_sql(q, conn, params=params)


def upcoming_matches(date_from: str, date_to: str) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(
            "SELECT * FROM matches WHERE status = 'scheduled' AND date >= ? AND date <= ? "
            "ORDER BY kickoff_utc",
            conn, params=(date_from, date_to),
        )


def scrape_progress() -> pd.DataFrame:
    """Completed-match counts by season/competition and detail status."""
    with get_conn() as conn:
        return pd.read_sql(
            "SELECT season, competition, details_status, COUNT(*) AS n FROM matches "
            "WHERE status = 'completed' GROUP BY season, competition, details_status "
            "ORDER BY season, competition",
            conn,
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _load(table: str, since: str = None, seasons: list = None) -> pd.DataFrame:
    q, params = f"SELECT * FROM {table} WHERE 1=1", []
    if since:
        q += " AND date >= ?"
        params.append(since)
    if seasons:
        q += f" AND season IN ({', '.join('?' for _ in seasons)})"
        params += list(seasons)
    with get_conn() as conn:
        return pd.read_sql(q, conn, params=params)


def load_player_match_stats(since: str = None, seasons: list = None) -> pd.DataFrame:
    return _load("player_match_stats", since, seasons)


def load_team_match_stats(since: str = None, seasons: list = None) -> pd.DataFrame:
    return _load("team_match_stats", since, seasons)


def load_player_unavailability(since: str = None, seasons: list = None) -> pd.DataFrame:
    return _load("player_unavailability", since, seasons)


# ---------------------------------------------------------------------------
# Predictions, metrics, drift
# ---------------------------------------------------------------------------

def save_predictions(df: pd.DataFrame):
    if df.empty:
        return
    with get_conn() as conn:
        _insert(conn, "predictions", df, PREDICTION_COLS)


def latest_predictions() -> pd.DataFrame:
    with get_conn() as conn:
        latest_ts = conn.execute("SELECT MAX(generated_at) FROM predictions").fetchone()[0]
        if latest_ts is None:
            return pd.DataFrame()
        return pd.read_sql(
            "SELECT * FROM predictions WHERE generated_at = ?", conn, params=(latest_ts,)
        )


def save_model_run(report: dict, trained_at: str):
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO model_runs (trained_at, report_json) VALUES (?, ?)",
                     (trained_at, json.dumps(report, default=str)))


def latest_model_report() -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT report_json FROM model_runs ORDER BY trained_at DESC LIMIT 1").fetchone()
    return json.loads(row[0]) if row else {}


def latest_predictions_before_kickoff() -> pd.DataFrame:
    """
    For each (match, player), the last prediction made before kickoff —
    what was actually "on the record" for that match. Used for the track record.
    """
    with get_conn() as conn:
        return pd.read_sql("""
            SELECT p.* FROM predictions p
            JOIN (SELECT match_id, player_id, MAX(generated_at) AS g FROM predictions
                  WHERE kickoff_utc IS NULL OR generated_at < kickoff_utc
                  GROUP BY match_id, player_id) last
              ON p.match_id = last.match_id AND p.player_id = last.player_id AND p.generated_at = last.g
        """, conn)


def posted_match_ids(kind: str = "picks") -> dict:
    """{match_id: lineup_type it was posted with}."""
    with get_conn() as conn:
        return dict(conn.execute("SELECT match_id, lineup_type FROM telegram_posts WHERE kind = ?",
                                 (kind,)).fetchall())


def record_telegram_post(match_id: str, lineup_type: str, message_ids: list, kind: str = "picks"):
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO telegram_posts (match_id, kind, lineup_type, posted_at, "
                     "message_ids) VALUES (?, ?, ?, ?, ?)",
                     (match_id, kind, lineup_type, _now(), json.dumps(message_ids)))


def save_model_metrics(row: dict):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO model_metrics
            (retrain_ts, mae_shots, mae_sot, n_train_rows, n_test_rows, holdout_from, holdout_to)
            VALUES (:retrain_ts, :mae_shots, :mae_sot, :n_train_rows, :n_test_rows, :holdout_from, :holdout_to)
        """, row)


def load_model_metrics() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM model_metrics ORDER BY retrain_ts", conn)


def save_drift_metrics(df: pd.DataFrame):
    if df.empty:
        return
    with get_conn() as conn:
        _insert(conn, "drift_metrics", df, ["retrain_ts", "feature_name", "psi_score", "flag"])


def load_drift_metrics() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM drift_metrics ORDER BY retrain_ts", conn)
