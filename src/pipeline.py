"""
Orchestration layer. Every entry point here is what app.py, the CLI
scripts and the GitHub Actions workflow actually call — scraper.py and
features.py stay "dumb" (no knowledge of the DB, no knowledge of what's
already been fetched).
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

import config
from src import db, scraper, features, train, evaluate, telegram


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
    upcoming = db.upcoming_matches(date_from, date_to)
    # A match FotMob still lists as "scheduled" may already have kicked off
    # (the fixture list is only refreshed when we ask) - never predict those.
    now = datetime.now(timezone.utc)
    return upcoming[pd.to_datetime(upcoming["kickoff_utc"], utc=True, format="ISO8601") > now].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

def _history():
    return db.load_player_match_stats(), db.load_team_match_stats()


def weekly_retrain() -> dict:
    """
    Rebuild features from every stored match, compare baseline / GLM /
    XGBoost on the most recent weeks, refit the best on everything, save.
    """
    db.init_db()
    players, teams = _history()
    if players.empty:
        return {"status": "no_data"}
    feat = features.build_features(players, teams)
    rows = features.training_rows(feat)
    report = train.train_and_save(rows)
    trained_at = train.load_bundle()["trained_at"]
    db.save_model_run(report, trained_at)
    return {"status": "ok", "trained_at": trained_at, "report": report}


def _fallback_starters(players: pd.DataFrame, team_id: int, unavailable: set) -> pd.DataFrame:
    """
    When FotMob has no lineup yet: players who started at least 3 of the
    team's last 5 matches, whose latest appearance was for this team, and
    who aren't listed as unavailable.
    """
    team_rows = players[players.team_id == team_id]
    last5 = team_rows.drop_duplicates("match_id").sort_values("date").tail(5).match_id
    starts = team_rows[team_rows.match_id.isin(last5) & (team_rows.started == 1)]
    counts = starts.groupby("player_id").size()
    latest = players.sort_values("date").groupby("player_id").tail(1).set_index("player_id")
    keep = [pid for pid, n in counts.items()
            if n >= 3 and latest.loc[pid, "team_id"] == team_id and pid not in unavailable]
    return latest.loc[keep, ["player_name", "position"]].reset_index().assign(team_id=team_id)


def expected_lineups(upcoming: pd.DataFrame, players: pd.DataFrame, session=None, log=print):
    """
    Expected starters for each upcoming fixture: FotMob's predicted (or
    confirmed) XI when published, else the fallback heuristic.
    Returns (player rows, team rows) shaped for features.build_features.
    """
    session = session or scraper._session()
    p_rows, t_rows = [], []
    for m in upcoming.to_dict("records"):
        try:
            lu = scraper.parse_upcoming_lineup(scraper.fetch_upcoming_details(m["match_id"], session), m)
        except scraper.BlockedError:
            raise
        except Exception as e:
            log(f"[lineups] {m['home_team']} v {m['away_team']}: {e} - using fallback")
            lu = {"starters": pd.DataFrame(), "unavailable": set(), "lineup_type": None}

        sides = ((m["home_team_id"], m["home_team"], m["away_team_id"], m["away_team"], 1),
                 (m["away_team_id"], m["away_team"], m["home_team_id"], m["home_team"], 0))
        for team_id, team, opp_id, opp, is_home in sides:
            team_id, opp_id = int(team_id), int(opp_id)
            starters = lu["starters"]
            starters = starters[starters.team_id == team_id] if not starters.empty else starters
            lineup_type = lu["lineup_type"]
            if starters.empty:
                starters, lineup_type = _fallback_starters(players, team_id, lu["unavailable"]), "estimated"
            base = {"match_id": m["match_id"], "date": m["date"], "competition": m["competition"],
                    "season": m["season"], "kickoff_utc": m["kickoff_utc"]}
            t_rows.append({**base, "team_id": team_id, "team": team, "opponent_id": opp_id,
                           "opponent": opp, "is_home": is_home})
            for s in starters.to_dict("records"):
                p_rows.append({**base, "player_id": int(s["player_id"]), "player_name": s["player_name"],
                               "team_id": team_id, "team": team, "opponent_id": opp_id,
                               "opponent": opp, "is_home": is_home, "started": 1,
                               "position": s.get("position"), "lineup_type": lineup_type})
    return pd.DataFrame(p_rows), pd.DataFrame(t_rows)


def generate_weekly_predictions(min_percentage: float = None, shot_thresholds=None,
                                sot_thresholds=None, refresh_fixtures: bool = True,
                                days_ahead: int = 7, log=print) -> pd.DataFrame:
    """
    For every expected starter in the next `days_ahead` days' fixtures,
    build their feature row and score it with the saved model. Returns the
    predictions and persists them (timestamped) to the `predictions` table.
    """
    db.init_db()
    if not os.path.exists(config.MODEL_BUNDLE_PATH):
        raise FileNotFoundError("No trained model yet - run a retrain first.")
    upcoming = upcoming_fixtures(days_ahead, refresh=refresh_fixtures)
    players, teams = _history()
    if upcoming.empty or players.empty:
        return pd.DataFrame()

    shot_thresholds = shot_thresholds or config.DEFAULT_SHOT_THRESHOLDS
    sot_thresholds = sot_thresholds or config.DEFAULT_SOT_THRESHOLDS

    up_players, up_teams = expected_lineups(upcoming, players, log=log)
    if up_players.empty:
        return pd.DataFrame()
    feat = features.build_features(players, teams, up_players, up_teams)
    rows = feat[(feat.is_future == 1) & feat.position.isin(features.POSITIONS)]
    if rows.empty:
        return pd.DataFrame()

    scored = train.predict(rows, shot_thresholds, sot_thresholds)
    scored["generated_at"] = _now()
    scored["sample_size_overall"] = scored["form_games"].astype(int)
    scored["sample_size_vs_opp"] = scored["h2h_player_n"].astype(int)
    db.save_predictions(scored[db.PREDICTION_COLS])
    return scored.sort_values(["kickoff_utc", "match_id", "pred_shots"], ascending=[True, True, False])


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

def publish_files(preds: pd.DataFrame) -> list:
    """
    Write the small files the deployed app reads (the database itself
    stays local). Returns the paths written.
    """
    out = []
    if preds is not None and not preds.empty:
        path = os.path.join(config.PUBLISHED_DIR, "predictions_latest.csv")
        preds[db.PREDICTION_COLS].to_csv(path, index=False)
        out.append(path)
    record = evaluate.track_record()
    path = os.path.join(config.PUBLISHED_DIR, "track_record.csv")
    record.to_csv(path, index=False)
    out.append(path)
    path = os.path.join(config.PUBLISHED_DIR, "model_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db.latest_model_report(), f, indent=1, default=str)
    out.append(path)
    return out


def post_to_telegram(preds: pd.DataFrame, min_percentage: float = None, within_hours: int = 24,
                     top_n: int = None, dry_run: bool = False, log=print) -> dict:
    """
    One daily digest: the top_n most confident players across fixtures
    kicking off within `within_hours`. Posts at most once per day, and a
    fixture already covered by an earlier digest is never included again.
    dry_run prints instead.
    """
    min_percentage = config.TELEGRAM_MIN_PROB if min_percentage is None else min_percentage
    top_n = config.TELEGRAM_TOP_N if top_n is None else top_n
    if preds is None or preds.empty:
        return {"posted": 0, "reason": "no predictions"}
    token, chat_id = telegram.credentials()
    if not dry_run and not (token and chat_id):
        return {"posted": 0, "reason": "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set (see src/telegram.py)"}

    now = datetime.now(timezone.utc)
    day_key = f"day:{now:%Y-%m-%d}"
    posted = db.posted_match_ids(kind="digest")
    if day_key in posted and not dry_run:
        return {"posted": 0, "reason": "today's digest was already posted"}

    kickoff = pd.to_datetime(preds["kickoff_utc"], utc=True, format="ISO8601")
    window = preds[(kickoff > now) & (kickoff <= now + timedelta(hours=within_hours))]
    window = window[~window.match_id.isin(posted)]
    title = f"Shots picks — {now:%a %d %b}"
    messages = telegram.build_digest(window, min_percentage, top_n, title)

    sent = 0
    for text, _ in messages:
        if dry_run:
            log("----- (dry run) -----\n" + text)
            continue
        msg_id = telegram.send_message(text, token, chat_id)
        sent += 1
    if sent:
        for mid in window.match_id.unique():
            lt = window.loc[window.match_id == mid, "lineup_type"].iloc[0]
            db.record_telegram_post(mid, lt, [], kind="digest")
        db.record_telegram_post(day_key, None, [], kind="digest")
    return {"posted": sent, "messages": len(messages), "dry_run": dry_run,
            "fixtures": int(window.match_id.nunique())}
