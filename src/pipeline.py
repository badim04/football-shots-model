"""
Orchestration layer. Every entry point here is what app.py, the CLI
scripts and the GitHub Actions workflow actually call — scraper.py and
features.py stay "dumb" (no knowledge of the DB, no knowledge of what's
already been fetched).
"""

import time
from datetime import datetime, timezone

import pandas as pd

import config
from src import db, scraper, features, train, drift


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def sync_fixtures(seasons: list = None, competitions: list = None,
                  session=None, log=print) -> dict:
    """
    Refresh the fixture list (played + upcoming) for each competition and
    season: one request per competition-season. A past season whose
    fixtures are already stored and fully decided is skipped — it can't
    change any more, so re-running the backfill doesn't re-ask for it.
    """
    db.init_db()
    seasons = seasons or [config.CURRENT_SEASON]
    competitions = competitions or list(config.COMPETITIONS)
    session = session or scraper._session()

    known = db.load_matches(seasons=seasons)
    synced, skipped = [], []
    for season in seasons:
        for comp in competitions:
            have = known[(known.season == season) & (known.competition == comp)]
            if (season != config.CURRENT_SEASON and not have.empty
                    and not have.status.isin(["scheduled", "live"]).any()):
                skipped.append((season, comp))
                continue
            fixtures = scraper.fetch_season_fixtures(comp, season, session=session)
            db.upsert_matches(fixtures)
            synced.append((season, comp, len(fixtures)))
            log(f"[fixtures] {season} {comp}: {len(fixtures)} matches "
                f"({(fixtures.status == 'completed').sum() if not fixtures.empty else 0} completed)")
    return {"synced": synced, "skipped_unchanged": skipped}


def scrape_pending_details(seasons: list = None, competitions: list = None,
                           limit: int = None, session=None, log=print) -> dict:
    """
    Fetch and store player/team stats for every completed match that
    doesn't have them yet. Each match is committed on its own, so this is
    safe to interrupt and re-run at any point — it resumes where it
    stopped. Stops early (without error) if FotMob starts refusing us.
    """
    db.init_db()
    session = session or scraper._session()
    todo = db.matches_needing_details(seasons, competitions)
    if limit:
        todo = todo.head(limit)

    counts = {"ok": 0, "no_stats": 0, "failed": 0}
    failures, blocked = [], None
    started = time.time()
    for i, m in enumerate(todo.to_dict("records"), start=1):
        try:
            details = scraper.fetch_match_details(m["match_id"], session=session)
            status = db.save_match_details(m["match_id"], scraper.parse_match_details(details, m))
            counts[status] += 1
        except scraper.BlockedError as e:
            blocked = str(e)
            log(f"[details] blocked - stopping. Progress is saved; re-run later to continue. ({e})")
            break
        except Exception as e:  # one bad match must not sink the run
            db.mark_details_failed(m["match_id"], f"{type(e).__name__}: {e}")
            counts["failed"] += 1
            failures.append((m["match_id"], str(e)))

        if i % 25 == 0 or i == len(todo):
            elapsed = time.time() - started
            eta_min = elapsed / i * (len(todo) - i) / 60
            log(f"[details] {i}/{len(todo)} - ok {counts['ok']}, no stats {counts['no_stats']}, "
                f"failed {counts['failed']} - ~{eta_min:.0f} min left")

    return {"matches_to_fetch": len(todo), **counts, "failures": failures, "blocked": blocked}


def reparse_from_cache(seasons: list = None, log=print) -> dict:
    """
    Rebuild the player/team tables from the raw JSON cache with zero
    network requests — for after a parser change, or to pick up a stat
    that wasn't being extracted before.
    """
    db.init_db()
    matches = db.load_matches(status="completed", seasons=seasons)
    counts = {"ok": 0, "no_stats": 0, "not_cached": 0, "failed": 0}
    for m in matches.to_dict("records"):
        raw = scraper.load_cached_details(m["match_id"])
        if raw is None:
            counts["not_cached"] += 1
            continue
        try:
            counts[db.save_match_details(m["match_id"], scraper.parse_match_details(raw, m))] += 1
        except Exception as e:
            db.mark_details_failed(m["match_id"], f"reparse {type(e).__name__}: {e}")
            counts["failed"] += 1
    log(f"[reparse] {counts}")
    return counts


def weekly_update(log=print) -> dict:
    """
    The scheduled data step: refresh this season's fixtures, then fetch
    stats for any newly completed match (plus any earlier failures).
    Scoped to the current season so an unfinished backfill never lands on
    the weekly job — the backfill runs locally (scripts/backfill.py).
    """
    fixtures = sync_fixtures([config.CURRENT_SEASON], log=log)
    details = scrape_pending_details([config.CURRENT_SEASON], log=log)
    return {"fixtures": fixtures, "details": details}


def upcoming_fixtures(days_ahead: int = 7, refresh: bool = True) -> pd.DataFrame:
    """Scheduled matches in the next N days (refreshing this season's fixture lists first)."""
    if refresh:
        sync_fixtures([config.CURRENT_SEASON], log=lambda *_: None)
    date_from, date_to = scraper.next_n_days_range(days_ahead)
    return db.upcoming_matches(date_from, date_to)


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

def weekly_retrain(recent_n_overall: int = None, recent_n_opponent: int = None) -> dict:
    """
    Full retrain on everything in the database: rebuild features for all
    completed matches, time-based train/test split, fit both models, run
    drift detection, persist model + metrics.
    """
    db.init_db()
    recent_n_overall = recent_n_overall or config.RECENT_N_OVERALL
    recent_n_opponent = recent_n_opponent or config.RECENT_N_OPPONENT

    raw = db.load_player_match_stats()
    if raw.empty:
        return {"status": "no_data"}

    feat_df = features.build_feature_table(raw, recent_n_overall, recent_n_opponent)
    metrics = train.train_and_save(feat_df)
    # Drift compares within the current season only — early weeks of THIS
    # season vs. its latest week (stage 3 replaces this with a calibration check).
    drift_report = drift.run_drift_check(feat_df[feat_df.season == config.CURRENT_SEASON])

    db.save_model_metrics(metrics)
    db.save_drift_metrics(drift_report)

    return {"status": "ok", "metrics": metrics, "drift": drift_report.to_dict("records")}


def generate_weekly_predictions(min_percentage: float = None,
                                shot_thresholds=None, sot_thresholds=None,
                                recent_n_overall: int = None,
                                recent_n_opponent: int = None,
                                refresh_fixtures: bool = True,
                                days_ahead: int = 7) -> pd.DataFrame:
    """
    For every player likely to feature in the next `days_ahead` days' fixtures, build
    their current feature row (using the opponent from their next match)
    and score it with the saved model. Returns the predictions DataFrame
    and also persists it (timestamped) to the `predictions` table.
    """
    db.init_db()
    upcoming = upcoming_fixtures(days_ahead, refresh=refresh_fixtures)
    if upcoming.empty:
        return pd.DataFrame()

    raw = db.load_player_match_stats()
    if raw.empty:
        return pd.DataFrame()

    recent_n_overall = recent_n_overall or config.RECENT_N_OVERALL
    recent_n_opponent = recent_n_opponent or config.RECENT_N_OPPONENT
    min_percentage = min_percentage if min_percentage is not None else config.DEFAULT_MIN_PERCENTAGE
    shot_thresholds = shot_thresholds or config.DEFAULT_SHOT_THRESHOLDS
    sot_thresholds = sot_thresholds or config.DEFAULT_SOT_THRESHOLDS

    pred_rows = features.build_upcoming_feature_rows(
        raw, upcoming, recent_n_overall, recent_n_opponent
    )
    if pred_rows.empty:
        return pd.DataFrame()

    scored = train.predict(pred_rows, shot_thresholds, sot_thresholds, min_percentage)
    scored["generated_at"] = _now()

    # prob_json carries every threshold probability; the eligible_* columns
    # are recomputed at display time from prob_json + the slider.
    db.save_predictions(scored[db.PREDICTION_COLS])
    return scored
