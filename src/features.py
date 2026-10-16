"""
Turns raw player_match_stats rows into a model-ready feature table.

Every feature here is computed strictly from matches BEFORE the row's own
date — a player's form going into a match never includes that match's own
result. This is what makes the later time-based train/test split honest;
getting this wrong (leaking same-match or future data into the features)
is the single most common way sports prediction models look good in
testing and fail in production.
"""

import numpy as np
import pandas as pd

import config


def _team_match_aggregates(raw: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (match_id, team): shots_for/against and sot_for/against,
    derived purely by summing player rows for that team vs. the opponent
    in the same match. This is the source of the opponent defense rating —
    no separate team-stats scrape needed.
    """
    per_team = (
        raw.groupby(["match_id", "team_id", "opponent_id", "date", "competition"])
        .agg(shots_for=("shots", "sum"), sot_for=("shots_on_target", "sum"))
        .reset_index()
    )
    # shots_against for `team` = shots_for by `opponent` in the same match
    merged = per_team.merge(
        per_team[["match_id", "team_id", "shots_for", "sot_for"]].rename(
            columns={"team_id": "opponent_id", "shots_for": "shots_against", "sot_for": "sot_against"}
        ),
        on=["match_id", "opponent_id"],
        how="left",
    )
    return merged


def _rolling_team_defense(team_agg: pd.DataFrame, recent_n: int) -> pd.DataFrame:
    """
    For each team, as of each date, the rolling average shots/SOT conceded
    over their last `recent_n` matches STRICTLY BEFORE that date.
    Returned as a lookup keyed by (team, date) so it can be joined onto
    both historical player rows (for training) and upcoming fixtures.
    """
    team_agg = team_agg.sort_values(["team_id", "date"]).reset_index(drop=True)
    out = []
    for team_id, grp in team_agg.groupby("team_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        shots_against_avg = grp["shots_against"].shift(1).rolling(recent_n, min_periods=1).mean()
        sot_against_avg = grp["sot_against"].shift(1).rolling(recent_n, min_periods=1).mean()
        n_games = grp["shots_against"].shift(1).expanding().count()
        out.append(pd.DataFrame({
            "team_id": team_id,
            "date": grp["date"],
            "opp_def_shots_conceded_avg": shots_against_avg,
            "opp_def_sot_conceded_avg": sot_against_avg,
            "opp_def_sample_size": n_games.clip(upper=recent_n),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["team_id", "date", "opp_def_shots_conceded_avg", "opp_def_sot_conceded_avg", "opp_def_sample_size"]
    )


def _league_normalize(team_defense: pd.DataFrame, team_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw shots-conceded into a league-relative defense rating
    (z-score against that competition's rolling average at the same point
    in time), so "tough defense" means the same thing in Ligue 1 as in
    the Champions League despite different scoring baselines.
    """
    comp_lookup = team_agg[["team_id", "date", "competition"]].drop_duplicates()
    merged = team_defense.merge(comp_lookup, on=["team_id", "date"], how="left")
    merged["opp_def_rating"] = merged.groupby("competition")["opp_def_shots_conceded_avg"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
    )
    # Lower shots conceded than league average => tougher defense => positive rating.
    merged["opp_def_rating"] = -1 * merged["opp_def_rating"].fillna(0.0)
    return merged


def _rolling_player_form(raw: pd.DataFrame, recent_n: int, group_cols: list, prefix: str) -> pd.DataFrame:
    """
    Generic rolling-form builder. group_cols = ["player_id"] for overall
    form, or ["player_id", "opponent_id"] for form vs. this specific opponent.
    Every value is computed from matches strictly before the row's date.
    """
    raw = raw.sort_values(group_cols + ["date"]).reset_index(drop=True)
    out_frames = []
    for keys, grp in raw.groupby(group_cols):
        grp = grp.sort_values("date").reset_index(drop=True)
        shots_prior = grp["shots"].shift(1)
        sot_prior = grp["shots_on_target"].shift(1)

        avg_shots = shots_prior.rolling(recent_n, min_periods=1).mean()
        avg_sot = sot_prior.rolling(recent_n, min_periods=1).mean()
        n_games = shots_prior.expanding().count().clip(upper=recent_n)

        frame = grp[["match_id"]].copy()
        for col in group_cols:
            frame[col] = grp[col].values
        frame["date"] = grp["date"].values
        frame[f"{prefix}_avg_shots"] = avg_shots.values
        frame[f"{prefix}_avg_sot"] = avg_sot.values
        frame[f"{prefix}_sample_size"] = n_games.values
        out_frames.append(frame)

    if not out_frames:
        cols = group_cols + ["match_id", "date", f"{prefix}_avg_shots", f"{prefix}_avg_sot", f"{prefix}_sample_size"]
        return pd.DataFrame(columns=cols)
    return pd.concat(out_frames, ignore_index=True)


def build_feature_table(raw: pd.DataFrame, recent_n_overall: int, recent_n_opponent: int) -> pd.DataFrame:
    """
    Build the full training table: one row per (match_id, player_id),
    with rolling overall form, rolling vs-opponent form, opponent defense
    rating, and the TARGET columns (shots, shots_on_target) for that match.
    """
    all_rows = raw.dropna(subset=["shots", "shots_on_target"]).copy()
    raw = all_rows[all_rows.minutes.fillna(0) >= config.MIN_MINUTES_TO_COUNT]
    if raw.empty:
        return pd.DataFrame()

    overall_form = _rolling_player_form(raw, recent_n_overall, ["player_id"], "overall")
    vs_opp_form = _rolling_player_form(raw, recent_n_opponent, ["player_id", "opponent_id"], "vs_opp")

    # Team totals use every player who played, substitutes included — a
    # defence conceded those shots whoever took them.
    team_agg = _team_match_aggregates(all_rows)
    team_defense = _rolling_team_defense(team_agg, recent_n_overall)
    team_defense = _league_normalize(team_defense, team_agg)

    df = raw.merge(overall_form, on=["match_id", "player_id", "date"], how="left")
    df = df.merge(
        vs_opp_form, on=["match_id", "player_id", "opponent_id", "date"], how="left"
    )
    df = df.merge(
        team_defense[["team_id", "date", "opp_def_shots_conceded_avg", "opp_def_sot_conceded_avg",
                       "opp_def_rating", "opp_def_sample_size"]].rename(columns={"team_id": "opponent_id"}),
        on=["opponent_id", "date"], how="left",
    )

    # First appearance of the season has no prior form — fill with 0 and
    # flag via sample_size=0 rather than dropping the row (games 1-2 for
    # every player would otherwise never train the model at all).
    fill_cols = [
        "overall_avg_shots", "overall_avg_sot", "overall_sample_size",
        "vs_opp_avg_shots", "vs_opp_avg_sot", "vs_opp_sample_size",
        "opp_def_shots_conceded_avg", "opp_def_sot_conceded_avg", "opp_def_rating", "opp_def_sample_size",
    ]
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df["target_shots"] = df["shots"]
    df["target_sot"] = df["shots_on_target"]
    return df


FEATURE_COLUMNS = [
    "overall_avg_shots", "overall_avg_sot", "overall_sample_size",
    "vs_opp_avg_shots", "vs_opp_avg_sot", "vs_opp_sample_size",
    "opp_def_shots_conceded_avg", "opp_def_sot_conceded_avg", "opp_def_rating",
    "is_home",
]


def build_upcoming_feature_rows(raw: pd.DataFrame, upcoming: pd.DataFrame,
                                 recent_n_overall: int, recent_n_opponent: int) -> pd.DataFrame:
    """
    For each upcoming fixture, build one feature row per "likely starter" —
    a player with a meaningful appearance rate over their team's recent
    matches. There is no confirmed-lineup data before kickoff, so this is
    a deliberate heuristic, not a guess dressed up as certainty: players
    who barely feature are excluded rather than silently predicted for.
    """
    all_rows = raw.dropna(subset=["shots", "shots_on_target"]).copy()
    raw = all_rows[all_rows.minutes.fillna(0) >= config.MIN_MINUTES_TO_COUNT]
    if raw.empty or upcoming.empty:
        return pd.DataFrame()

    team_agg = _team_match_aggregates(all_rows)
    team_defense = _rolling_team_defense(team_agg, recent_n_overall)
    team_defense = _league_normalize(team_defense, team_agg)
    # Use each team's most recent defense rating as of "today" for the opponent side.
    latest_defense = (
        team_defense.sort_values("date").groupby("team_id").tail(1)
        .set_index("team_id")[["opp_def_shots_conceded_avg", "opp_def_sot_conceded_avg",
                             "opp_def_rating", "opp_def_sample_size"]]
    )

    rows = []
    for _, fx in upcoming.iterrows():
        for team_id, team, opp_id, opponent, is_home in [
            (fx.home_team_id, fx.home_team, fx.away_team_id, fx.away_team, 1),
            (fx.away_team_id, fx.away_team, fx.home_team_id, fx.home_team, 0),
        ]:
            squad = raw[raw.team_id == team_id]
            if squad.empty:
                continue
            recent_matches = squad.drop_duplicates("match_id").sort_values("date").tail(recent_n_overall * 2)
            recent_match_ids = set(recent_matches.match_id)
            appearance_rate = (
                squad[squad.match_id.isin(recent_match_ids)]
                .groupby("player_id")
                .match_id.nunique() / max(len(recent_match_ids), 1)
            )
            likely_starters = appearance_rate[appearance_rate >= 0.5].index.tolist()

            for player_id in likely_starters:
                # Form follows the player, not the club — include matches
                # for a previous team, so a new signing isn't a blank slate.
                p_hist = raw[raw.player_id == player_id].sort_values("date")
                overall_recent = p_hist.tail(recent_n_overall)
                vs_opp_hist = p_hist[p_hist.opponent_id == opp_id].tail(recent_n_opponent)

                def_row = latest_defense.loc[opp_id] if opp_id in latest_defense.index else None

                rows.append({
                    "match_id": fx.match_id,
                    "date": fx.date,
                    "competition": fx.competition,
                    "season": fx.season,
                    "player_id": player_id,
                    "player_name": p_hist.player_name.iloc[-1],
                    "team_id": team_id,
                    "team": team,
                    "opponent_id": opp_id,
                    "opponent": opponent,
                    "is_home": is_home,
                    "overall_avg_shots": overall_recent.shots.mean() if len(overall_recent) else 0,
                    "overall_avg_sot": overall_recent.shots_on_target.mean() if len(overall_recent) else 0,
                    "overall_sample_size": len(overall_recent),
                    "vs_opp_avg_shots": vs_opp_hist.shots.mean() if len(vs_opp_hist) else 0,
                    "vs_opp_avg_sot": vs_opp_hist.shots_on_target.mean() if len(vs_opp_hist) else 0,
                    "vs_opp_sample_size": len(vs_opp_hist),
                    "opp_def_shots_conceded_avg": def_row["opp_def_shots_conceded_avg"] if def_row is not None else 0,
                    "opp_def_sot_conceded_avg": def_row["opp_def_sot_conceded_avg"] if def_row is not None else 0,
                    "opp_def_rating": def_row["opp_def_rating"] if def_row is not None else 0,
                    "opp_def_sample_size": def_row["opp_def_sample_size"] if def_row is not None else 0,
                })

    return pd.DataFrame(rows)
