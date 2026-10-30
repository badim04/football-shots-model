"""
Turns stored match history into a model-ready feature table.

One function, `build_features`, serves both training and prediction:
upcoming fixtures are appended to the history as rows with no result yet,
and every feature is computed from rows strictly BEFORE each row. That
guarantees a prediction sees exactly the kind of inputs the model was
trained on — the most common way sports models look good in testing and
fail live is building the two differently, or leaking the match itself
into its own features.

Feature groups (see FEATURE_GROUPS; scripts/evaluate.py tests each one):

- form          The player's last few games, per 90 minutes, weighted toward
                the most recent (half-life FORM_HALFLIFE_GAMES): shots, SOT,
                xG, touches in the box, plus recent minutes and starts.
- baseline      What the player normally does (a slower average, pulled
                toward his position's average while history is thin), his
                share of the team's shots, and penalty duty.
- h2h           Head-to-head across seasons: how this team's shot volume vs
                this opponent compares with its usual level, the same for
                the player, how many meetings that rests on, and a derby flag.
- team          Team attack form, opponent defensive form relative to the
                competition, opponent possession/attack (game state), home.
- circumstances Rest days (both sides), a European game just before, days
                since the player last played, games missed (injury return),
                games under the current coach.
"""

import numpy as np
import pandas as pd

import config

EUROPEAN = {"Champions League", "Europa League", "Conference League"}
POSITIONS = ["FWD", "MID", "DEF"]   # goalkeepers are excluded from training and predictions
FORM_PRIOR_MINUTES = 30             # how much "normal rate" is mixed into recent form

FEATURE_GROUPS = {
    "form": [
        "form_shots_p90", "form_sot_p90", "form_xg_p90", "form_box_p90",
        "form_minutes", "form_starts", "form_games", "form_vs_baseline",
    ],
    "baseline": [
        "baseline_shots_p90", "baseline_sot_p90", "shot_share", "pen_rate",
        "pos_FWD", "pos_MID", "pos_DEF",
    ],
    "h2h": [
        "h2h_team_ratio", "h2h_team_n", "h2h_team_days",
        "h2h_player_ratio", "h2h_player_n", "is_derby",
    ],
    "team": [
        "team_shots_for", "team_xg_for", "team_possession",
        "opp_shots_against_rel", "opp_xg_against", "opp_shots_for", "opp_possession",
        "league_avg_shots", "is_home", "is_europe",
    ],
    "circumstances": [
        "rest_days", "opp_rest_days", "prev_europe", "days_since_last_app",
        "last_minutes", "missed_games", "coach_games",
    ],
}
FEATURE_COLUMNS = [c for cols in FEATURE_GROUPS.values() for c in cols]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prior_ewm(df: pd.DataFrame, key, col: str, halflife: float) -> pd.Series:
    """Exponentially weighted mean of `col` over each group's PRIOR rows only."""
    prev = df.groupby(key, sort=False)[col].shift()
    out = (prev.groupby([df[k] for k in ([key] if isinstance(key, str) else key)], sort=False)
           .ewm(halflife=halflife, ignore_na=True).mean())
    return out.reset_index(level=list(range(out.index.nlevels - 1)), drop=True).reindex(df.index)


def _prior_cumsum(df: pd.DataFrame, key, col: str) -> pd.Series:
    """Sum of `col` over each group's prior rows (current row excluded)."""
    v = df[col].fillna(0)
    return v.groupby([df[k] for k in ([key] if isinstance(key, str) else key)], sort=False).cumsum() - v


def _prior_count(df: pd.DataFrame, key, col: str) -> pd.Series:
    """How many prior rows in the group have a value for `col` (i.e. were played)."""
    has = df[col].notna().astype(int)
    return has.groupby([df[k] for k in ([key] if isinstance(key, str) else key)], sort=False).cumsum() - has


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((b > 0) & np.isfinite(b), a / b, np.nan)


def _derby_set():
    return {frozenset(p) for p in config.DERBIES}


# ---------------------------------------------------------------------------
# Team-level features (one row per team per match)
# ---------------------------------------------------------------------------

def _team_features(teams: pd.DataFrame) -> pd.DataFrame:
    t = teams.copy()
    t["date_dt"] = pd.to_datetime(t["date"])
    t = t.sort_values(["team_id", "date_dt", "match_id"]).reset_index(drop=True)
    g = t.groupby("team_id", sort=False)
    hl = config.TEAM_FORM_HALFLIFE_GAMES

    for col in ["shots_for", "shots_against", "xg_for", "xg_against", "possession"]:
        t[f"team_{col}"] = _prior_ewm(t, "team_id", col, hl)
    t["team_shots_for_long"] = _prior_ewm(t, "team_id", "shots_for", config.LONG_RUN_HALFLIFE_GAMES)
    t["team_idx"] = g.cumcount()

    prev_date = g["date_dt"].shift()
    t["rest_days"] = (t["date_dt"] - prev_date).dt.days.clip(upper=14).fillna(14)
    t["prev_europe"] = (g["competition"].shift().isin(EUROPEAN) & (t["rest_days"] <= 4)).astype(int)

    # Games under the coach of the team's most recent match (a new-manager effect).
    coach = g["coach_id"].ffill()
    run = (coach != coach.groupby(t["team_id"]).shift()).groupby(t["team_id"]).cumsum()
    streak = t.groupby([t["team_id"], run]).cumcount() + 1
    t["coach_games"] = streak.groupby(t["team_id"]).shift().clip(upper=20).fillna(1)

    # Competition-season average shots per team-match, from earlier dates
    # only, lightly shrunk toward a typical value early in a season.
    t = t.sort_values(["competition", "season", "date_dt"]).reset_index(drop=True)
    daily = (t.groupby(["competition", "season", "date_dt"])["shots_for"]
             .agg(["sum", "count"]).reset_index())
    daily[["cum_sum", "cum_n"]] = (daily.groupby(["competition", "season"])[["sum", "count"]]
                                   .cumsum().values - daily[["sum", "count"]].values)
    typical = t["shots_for"].mean() if t["shots_for"].notna().any() else 12.5
    daily["league_avg_shots"] = (daily["cum_sum"] + typical * 20) / (daily["cum_n"] + 20)
    t = t.merge(daily[["competition", "season", "date_dt", "league_avg_shots"]],
                on=["competition", "season", "date_dt"], how="left")

    # Head-to-head at team level: each meeting's shots relative to the
    # team's usual level at the time, averaged over recent meetings.
    t = t.sort_values(["team_id", "opponent_id", "date_dt", "match_id"]).reset_index(drop=True)
    t["meeting_ratio"] = _safe_div(t["shots_for"], t["team_shots_for_long"])
    pair = ["team_id", "opponent_id"]
    ratio = _prior_ewm(t, pair, "meeting_ratio", config.H2H_HALFLIFE_MEETINGS)
    n = _prior_count(t, pair, "meeting_ratio")
    k = config.H2H_SHRINK_MEETINGS
    t["h2h_team_ratio"] = (n * ratio.fillna(1.0) + k) / (n + k)
    t["h2h_team_n"] = n.clip(upper=10)
    last_meeting = t.groupby(pair, sort=False)["date_dt"].shift()
    t["h2h_team_days"] = (t["date_dt"] - last_meeting).dt.days.clip(upper=1500).fillna(1500)
    return t


# ---------------------------------------------------------------------------
# The full table
# ---------------------------------------------------------------------------

def build_features(players: pd.DataFrame, teams: pd.DataFrame,
                   upcoming_players: pd.DataFrame = None,
                   upcoming_teams: pd.DataFrame = None) -> pd.DataFrame:
    """
    players/teams: stored history (player_match_stats / team_match_stats).
    upcoming_players: one row per expected starter in an upcoming fixture
        (match_id, date, competition, season, player_id, player_name,
        team_id, team, opponent_id, opponent, is_home, position, started=1).
    upcoming_teams: one row per team per upcoming fixture.

    Returns one row per player-match (history + upcoming) with every
    FEATURE_COLUMNS feature, `is_future`, and the targets (NaN for upcoming).
    """
    if players.empty or teams.empty:
        return pd.DataFrame()

    hist_p = players.assign(is_future=0)
    hist_t = teams.assign(is_future=0)
    if upcoming_players is not None and not upcoming_players.empty:
        hist_p = pd.concat([hist_p, upcoming_players.assign(is_future=1)], ignore_index=True)
    if upcoming_teams is not None and not upcoming_teams.empty:
        hist_t = pd.concat([hist_t, upcoming_teams.assign(is_future=1)], ignore_index=True)

    t = _team_features(hist_t)

    p = hist_p.copy()
    p["date_dt"] = pd.to_datetime(p["date"])
    p = p.merge(t[["match_id", "team_id", "team_idx", "shots_for"]].rename(
        columns={"shots_for": "team_shots_in_match"}), on=["match_id", "team_id"], how="left")
    p = p.sort_values(["player_id", "date_dt", "match_id"]).reset_index(drop=True)
    g = p.groupby("player_id", sort=False)

    # Position: the stored usual position, carried forward for upcoming rows without one.
    p["position"] = g["position"].ffill()

    # --- form (per 90, recency-weighted) ---
    hl = config.FORM_HALFLIFE_GAMES
    p["team_shots_x_min"] = p["team_shots_in_match"] * p["minutes"] / 90
    ew = {c: _prior_ewm(p, "player_id", c, hl)
          for c in ["shots", "shots_on_target", "xg", "touches_opp_box", "minutes", "started"]}
    p["form_minutes"] = ew["minutes"]
    p["form_starts"] = ew["started"]
    p["form_games"] = _prior_count(p, "player_id", "minutes").clip(upper=7)

    # --- baseline: slower average, shrunk toward the position's average ---
    played = p[(p.is_future == 0) & (p.minutes > 0)]
    pos_rate = (played.groupby("position")[["shots", "shots_on_target"]].sum()
                .div(played.groupby("position")["minutes"].sum(), axis=0) * 90)
    overall = played[["shots", "shots_on_target"]].sum() / max(played["minutes"].sum(), 1) * 90
    long_hl = config.LONG_RUN_HALFLIFE_GAMES
    long_ew = {c: _prior_ewm(p, "player_id", c, long_hl) for c in ["shots", "shots_on_target", "minutes"]}
    cum_minutes = _prior_cumsum(p, "player_id", "minutes").clip(upper=1800)
    w = cum_minutes / (cum_minutes + config.BASELINE_PRIOR_MINUTES)
    for name, col in [("shots", "shots"), ("sot", "shots_on_target")]:
        prior = p["position"].map(pos_rate[col]).fillna(overall[col])
        long_rate = pd.Series(_safe_div(long_ew[col], long_ew["minutes"]) * 90, index=p.index)
        p[f"baseline_{name}_p90"] = w * long_rate.fillna(prior) + (1 - w) * prior

    # Form per 90, with FORM_PRIOR_MINUTES of the player's normal rate mixed
    # in — so one shot in a 5-minute cameo reads as "a bit up", not "90 per 90".
    pm = FORM_PRIOR_MINUTES
    mins = ew["minutes"].fillna(0)
    for name, num, normal in [
        ("shots", ew["shots"], p["baseline_shots_p90"]),
        ("sot", ew["shots_on_target"], p["baseline_sot_p90"]),
        ("xg", ew["xg"], p["baseline_shots_p90"] * 0.11),
        ("box", ew["touches_opp_box"], p["baseline_shots_p90"] * 2.5),
    ]:
        p[f"form_{name}_p90"] = (num.fillna(0) + normal * pm / 90) / (mins + pm) * 90
    p["form_minutes"] = p["form_minutes"].fillna(60)
    p["form_starts"] = p["form_starts"].fillna(0.5)
    p["form_vs_baseline"] = p["form_shots_p90"] - p["baseline_shots_p90"]
    p["shot_share"] = pd.Series(_safe_div(_prior_ewm(p, "player_id", "shots", long_hl),
                                          _prior_ewm(p, "player_id", "team_shots_x_min", long_hl)),
                                index=p.index).fillna(0.0).clip(upper=1.0)
    p["pen_rate"] = _prior_ewm(p, "player_id", "penalty_shots", long_hl).fillna(0.0)
    for pos in POSITIONS:
        p[f"pos_{pos}"] = (p["position"] == pos).astype(int)

    # --- circumstances (player side) ---
    prev_date = g["date_dt"].shift()
    p["days_since_last_app"] = (p["date_dt"] - prev_date).dt.days.clip(upper=60).fillna(60)
    p["last_minutes"] = g["minutes"].shift().fillna(0)
    prev_team, prev_idx = g["team_id"].shift(), g["team_idx"].shift()
    p["missed_games"] = np.where(prev_team == p["team_id"],
                                 (p["team_idx"] - prev_idx - 1).clip(lower=0, upper=10), 0)
    p["missed_games"] = pd.Series(p["missed_games"]).fillna(0)

    # --- head-to-head, player level: shots vs this opponent relative to what
    #     his baseline would have expected in those same minutes ---
    p = p.sort_values(["player_id", "opponent_id", "date_dt", "match_id"]).reset_index(drop=True)
    p["expected_shots"] = p["baseline_shots_p90"] * p["minutes"] / 90
    pair = ["player_id", "opponent_id"]
    cum_shots = _prior_cumsum(p, pair, "shots")
    cum_exp = _prior_cumsum(p, pair, "expected_shots")
    k = config.H2H_SHRINK_MEETINGS
    p["h2h_player_ratio"] = (cum_shots + k) / (cum_exp + k)
    p["h2h_player_n"] = _prior_count(p, pair, "minutes").clip(upper=10)

    derbies = _derby_set()
    p["is_derby"] = [int(frozenset((a, b)) in derbies) for a, b in zip(p["team"], p["opponent"])]
    p["is_europe"] = p["competition"].isin(EUROPEAN).astype(int)

    # --- team + opponent context ---
    own = t[["match_id", "team_id", "team_shots_for", "team_xg_for", "team_possession",
             "league_avg_shots", "rest_days", "prev_europe", "coach_games",
             "h2h_team_ratio", "h2h_team_n", "h2h_team_days"]]
    opp = t[["match_id", "team_id", "team_shots_against", "team_xg_against",
             "team_shots_for", "team_possession", "rest_days"]].rename(columns={
        "team_id": "opponent_id", "team_shots_against": "opp_shots_against",
        "team_xg_against": "opp_xg_against", "team_shots_for": "opp_shots_for",
        "team_possession": "opp_possession", "rest_days": "opp_rest_days"})
    p = p.merge(own, on=["match_id", "team_id"], how="left")
    p = p.merge(opp, on=["match_id", "opponent_id"], how="left")
    p["opp_shots_against_rel"] = _safe_div(p["opp_shots_against"], p["league_avg_shots"])

    # Teams with no prior matches in the data (e.g. newly promoted, first
    # game) get typical values rather than gaps.
    defaults = {
        "team_shots_for": p["league_avg_shots"], "team_xg_for": 1.3, "team_possession": 50.0,
        "opp_shots_against": p["league_avg_shots"], "opp_shots_against_rel": 1.0,
        "opp_xg_against": 1.3, "opp_shots_for": p["league_avg_shots"], "opp_possession": 50.0,
        "league_avg_shots": 12.5, "rest_days": 14, "opp_rest_days": 14, "prev_europe": 0,
        "coach_games": 1, "h2h_team_ratio": 1.0, "h2h_team_n": 0, "h2h_team_days": 1500,
    }
    for col, val in defaults.items():
        p[col] = p[col].fillna(val)

    p["target_shots"] = p["shots"]
    p["target_sot"] = p["shots_on_target"]
    return p.sort_values(["date_dt", "match_id", "team_id", "player_id"]).reset_index(drop=True)


def training_rows(feat: pd.DataFrame) -> pd.DataFrame:
    """
    The rows a model learns from: completed matches, outfield players who
    STARTED. Predictions are made for expected starters, so the model
    learns "shots given he starts"; how long starters usually stay on is
    in the features (form_minutes, form_starts).
    """
    if feat.empty:
        return feat
    return feat[(feat.is_future == 0) & (feat.started == 1) & (feat.position.isin(POSITIONS))
                & feat.target_shots.notna()].copy()
