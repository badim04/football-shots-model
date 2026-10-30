"""
FotMob data client.

Why FotMob: FBref lost its Opta licence in January 2026 (per-player shots
and shots on target went with it) and sits behind a Cloudflare challenge
that a plain HTTP client can't pass. FotMob's own site is fed by public
JSON endpoints that carry exactly what this model needs — per-player
minutes, shots, shots on target and xG, starters vs substitutes, injured
players, the coach, and team possession — with stable player and team IDs,
for every competition in scope including the European ones.

These endpoints are unofficial and undocumented. They can change without
notice, which is why parsing is kept in one place (`parse_match_details`)
and every raw response is cached so a parser fix never needs a re-fetch.

Design choices, and why:

- Two endpoints only. `leagues?id=&season=` returns a competition's full
  season fixture list in one request; `matchDetails?matchId=` returns
  everything about one match. A weekly update is ~8 fixture-list requests
  plus one request per newly finished match.
- Every finished match is fetched once, ever: the raw JSON is gzipped to
  `data/raw_cache/` and never re-requested.
- Politeness means a randomized delay between requests and backing off on
  429/5xx. A 403 means we've been blocked, and the right response is to
  stop and let the next run pick up where this one left off — not to route
  around it — so it raises `BlockedError` instead of retrying.
"""

import gzip
import json
import os
import random
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config

BASE_URL = "https://www.fotmob.com/api/data"

# FotMob's `usualPosition` / `usualPlayingPositionId` codes.
POSITION_LABELS = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}


class BlockedError(RuntimeError):
    """FotMob returned 403 — stop this run rather than keep knocking."""


# ---------------------------------------------------------------------------
# HTTP layer: session, delay, retry/backoff
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": config.USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def _get_json(session: requests.Session, path: str, params: dict) -> dict:
    """GET one FotMob endpoint with a polite delay and retry/backoff."""
    time.sleep(random.uniform(config.MIN_DELAY_SECONDS, config.MAX_DELAY_SECONDS))
    url = f"{BASE_URL}/{path}"

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            status, detail = None, str(e)
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 403:
                raise BlockedError(f"403 from {resp.url} — stopping this run")
            if resp.status_code not in (429, 500, 502, 503, 504):
                resp.raise_for_status()
            status, detail = resp.status_code, resp.url

        wait = config.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 5)
        print(f"[scraper] {status or 'network error'} on {detail} - backing off {wait:.0f}s "
              f"(attempt {attempt}/{config.MAX_RETRIES})")
        time.sleep(wait)

    raise RuntimeError(f"Failed to fetch {url} {params} after {config.MAX_RETRIES} attempts")


def _raw_path(match_id: str) -> str:
    return os.path.join(config.RAW_CACHE_DIR, f"{match_id}.json.gz")


def load_cached_details(match_id: str):
    """The cached raw matchDetails JSON for a match, or None if never fetched."""
    path = _raw_path(match_id)
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1: season fixture lists
# ---------------------------------------------------------------------------

def _status(st: dict) -> str:
    if st.get("cancelled"):
        return "cancelled"
    if st.get("awarded"):
        return "awarded"      # decided off the pitch — no player stats to learn from
    if st.get("finished"):
        return "completed"
    if st.get("started"):
        return "live"
    return "scheduled"


def _score(score_str):
    try:
        home, away = (int(x) for x in str(score_str).split("-"))
        return home, away
    except (TypeError, ValueError):
        return None, None


def fetch_season_fixtures(competition: str, season: str,
                          session: requests.Session = None) -> pd.DataFrame:
    """
    One competition's full fixture list for one season (played and
    upcoming), one request. Columns match the `matches` table.
    """
    session = session or _session()
    league_id = config.COMPETITIONS[competition]
    data = _get_json(session, "leagues", {"id": league_id, "season": season})
    selected = data.get("details", {}).get("selectedSeason")
    if selected and selected != season:
        raise RuntimeError(f"Asked FotMob for {competition} {season}, got season {selected}")

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for m in data.get("fixtures", {}).get("allMatches", []):
        st = m.get("status", {})
        status = _status(st)
        home_score, away_score = _score(st.get("scoreStr")) if status == "completed" else (None, None)
        kickoff = st.get("utcTime") or ""
        rows.append({
            "match_id": str(m["id"]),
            "season": season,
            "competition": competition,
            "league_id": league_id,
            "round": str(m.get("round") or ""),
            "date": kickoff[:10],
            "kickoff_utc": kickoff,
            "home_team_id": int(m["home"]["id"]),
            "home_team": m["home"]["name"],
            "away_team_id": int(m["away"]["id"]),
            "away_team": m["away"]["name"],
            "home_score": home_score,
            "away_score": away_score,
            "status": status,
            "fixtures_synced_at": now,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: per-match details
# ---------------------------------------------------------------------------

def fetch_match_details(match_id: str, session: requests.Session = None) -> dict:
    """
    Raw matchDetails JSON for one match. Served from the cache when we
    already have it; a freshly fetched match is cached forever once it's
    finished AND has player stats (stats can lag the final whistle, and an
    empty payload must not be frozen into the cache).
    """
    cached = load_cached_details(match_id)
    if cached is not None:
        return cached

    session = session or _session()
    data = _get_json(session, "matchDetails", {"matchId": match_id})
    if data.get("general", {}).get("finished") and (data.get("content") or {}).get("playerStats"):
        tmp = _raw_path(match_id) + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, _raw_path(match_id))   # never leave a half-written cache file
    return data


def _num(v):
    """FotMob mixes ints, floats, numeric strings and '45%'-style strings."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().rstrip("%"))
    except ValueError:
        return None


def _flatten_player_stats(p: dict) -> dict:
    """{stat_key: value} across every stats section for one player."""
    out = {}
    for section in p.get("stats") or []:
        for stat in (section.get("stats") or {}).values():
            key = stat.get("key")
            if key:
                out[key] = (stat.get("stat") or {}).get("value")
    return out


def _flatten_team_stats(content: dict) -> dict:
    """{stat_key: (home_value, away_value)} from the full-match team stats."""
    out = {}
    periods = (content.get("stats") or {}).get("Periods") or {}
    for section in (periods.get("All") or {}).get("stats") or []:
        for stat in section.get("stats") or []:
            vals = stat.get("stats")
            if stat.get("key") and isinstance(vals, list) and len(vals) == 2:
                out[stat["key"]] = (_num(vals[0]), _num(vals[1]))
    return out


def parse_match_details(details: dict, match: dict) -> dict:
    """
    Turn one raw matchDetails payload into rows for the three per-match
    tables. `match` is this match's row from the `matches` table (a dict).

    Returns {"players": DataFrame, "teams": DataFrame, "unavailable": DataFrame}.
    Players who didn't get on the pitch are left out — they carry no
    information about shots — but every player who did play is kept,
    however few minutes; the minutes cut-off is applied later, in features.
    """
    content = details.get("content") or {}
    lineup = content.get("lineup") or {}
    home_id, away_id = int(match["home_team_id"]), int(match["away_team_id"])
    names = {home_id: match["home_team"], away_id: match["away_team"]}
    base = {"match_id": str(match["match_id"]), "date": match["date"],
            "competition": match["competition"], "season": match["season"]}

    # --- lineups: who started, formation, coach, and who was unavailable ---
    starters, lineup_by_team = set(), {}
    for side in ("homeTeam", "awayTeam"):
        t = lineup.get(side) or {}
        if "id" in t:
            lineup_by_team[int(t["id"])] = t
            starters.update(int(s["id"]) for s in t.get("starters") or [])
    have_lineups = bool(starters)

    # --- penalties per player, from the shot map (a penalty is a shot too,
    #     so knowing who takes them matters for the shots target) ---
    penalties = {}
    for shot in (content.get("shotmap") or {}).get("shots") or []:
        if shot.get("situation") == "Penalty" and not shot.get("isOwnGoal"):
            pid = int(shot["playerId"])
            penalties[pid] = penalties.get(pid, 0) + 1

    players = []
    for p in (content.get("playerStats") or {}).values():
        s = _flatten_player_stats(p)
        minutes = _num(s.get("minutes_played"))
        if not minutes:
            continue
        pid, team_id = int(p["id"]), int(p["teamId"])
        if team_id not in names:
            continue
        opp_id = away_id if team_id == home_id else home_id
        # FotMob drops a stat from a player's list when it's zero, so a
        # missing shots key means 0 shots for someone who played.
        shots = _num(s.get("total_shots")) or 0
        sot = _num(s.get("ShotsOnTarget")) or 0
        usual = p.get("usualPosition")
        players.append({
            **base,
            "player_id": pid,
            "player_name": p.get("name"),
            "team_id": team_id,
            "team": names[team_id],
            "opponent_id": opp_id,
            "opponent": names[opp_id],
            "is_home": int(team_id == home_id),
            "started": int(pid in starters) if have_lineups else None,
            "position": POSITION_LABELS.get(usual) if not p.get("isGoalkeeper") else "GK",
            "position_id": p.get("positionId"),
            "minutes": int(minutes),
            "shots": int(shots),
            "shots_on_target": int(sot),
            "penalty_shots": penalties.get(pid, 0),
            "goals": int(_num(s.get("goals")) or 0),
            "xg": _num(s.get("expected_goals")) or 0.0,
            "npxg": _num(s.get("expected_goals_non_penalty")),
            "touches_opp_box": _num(s.get("touches_opp_box")),
            "rating": _num(s.get("rating_title")),
        })

    team_stats = _flatten_team_stats(content)
    teams = []
    for team_id, opp_id, idx in ((home_id, away_id, 0), (away_id, home_id, 1)):
        t = lineup_by_team.get(team_id, {})
        coach = t.get("coach") or {}

        def stat(key):
            v = team_stats.get(key)
            return v[idx] if v else None

        def stat_against(key):
            v = team_stats.get(key)
            return v[1 - idx] if v else None

        teams.append({
            **base,
            "team_id": team_id,
            "team": names[team_id],
            "opponent_id": opp_id,
            "opponent": names[opp_id],
            "is_home": int(idx == 0),
            "goals_for": match.get("home_score") if idx == 0 else match.get("away_score"),
            "goals_against": match.get("away_score") if idx == 0 else match.get("home_score"),
            "shots_for": stat("total_shots"),
            "shots_against": stat_against("total_shots"),
            "sot_for": stat("ShotsOnTarget"),
            "sot_against": stat_against("ShotsOnTarget"),
            "xg_for": stat("expected_goals"),
            "xg_against": stat_against("expected_goals"),
            "possession": stat("BallPossesion"),   # sic — FotMob's key
            "touches_opp_box": stat("touches_opp_box"),
            "big_chances": stat("big_chance"),
            "formation": t.get("formation"),
            "coach_id": coach.get("id"),
            "coach_name": coach.get("name"),
        })

    unavailable = []
    for team_id, t in lineup_by_team.items():
        for u in t.get("unavailable") or []:
            info = u.get("unavailability") or {}
            unavailable.append({
                **base,
                "team_id": team_id,
                "player_id": int(u["id"]),
                "player_name": u.get("name"),
                "reason": info.get("type"),
                "expected_return": info.get("expectedReturn"),
            })

    return {
        "players": pd.DataFrame(players),
        "teams": pd.DataFrame(teams),
        "unavailable": pd.DataFrame(unavailable),
    }


def fetch_upcoming_details(match_id: str, session: requests.Session = None) -> dict:
    """matchDetails for a match that hasn't finished — never cached (lineups change)."""
    return _get_json(session or _session(), "matchDetails", {"matchId": match_id})


def parse_upcoming_lineup(details: dict, match: dict) -> dict:
    """
    FotMob's expected starting XI for an upcoming match, and who's out.
    `lineup_type` is "predicted" (FotMob's forecast, days ahead) or the
    confirmed team sheet once it's published (~1 hour before kickoff).

    Returns {"starters": DataFrame, "unavailable": set of player_ids, "lineup_type": str|None}.
    """
    lineup = (details.get("content") or {}).get("lineup") or {}
    lineup_type = lineup.get("lineupType")
    home_id, away_id = int(match["home_team_id"]), int(match["away_team_id"])
    rows, out = [], set()
    for side in ("homeTeam", "awayTeam"):
        t = lineup.get(side) or {}
        if "id" not in t:
            continue
        team_id = int(t["id"])
        if team_id not in (home_id, away_id):
            continue
        out.update(int(u["id"]) for u in t.get("unavailable") or [])
        for s in t.get("starters") or []:
            usual = s.get("usualPlayingPositionId")
            rows.append({
                "player_id": int(s["id"]),
                "player_name": s.get("name"),
                "team_id": team_id,
                "position": POSITION_LABELS.get(usual),
            })
    return {"starters": pd.DataFrame(rows), "unavailable": out,
            "lineup_type": ("confirmed" if lineup_type and lineup_type != "predicted" else lineup_type)
            if rows else None}


# ---------------------------------------------------------------------------
# Date-range helpers
# ---------------------------------------------------------------------------

def _today():
    return datetime.now(timezone.utc).date()


def next_n_days_range(n: int = 7, start_date: str = None) -> tuple:
    start = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else _today()
    return start.isoformat(), (start + timedelta(days=n)).isoformat()
