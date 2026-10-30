"""
Dev tool: a sanity check that storage, feature engineering, model
comparison/training and prediction all run end-to-end, on synthetic data.
Uses its own throwaway database and model file, so it never touches your
real backfilled data in data/matches.db or models/.
"""
import os
import random
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_tmp = tempfile.mkdtemp(prefix="shots_smoke_")
os.environ["SHOTS_DB_PATH"] = os.path.join(_tmp, "smoke.db")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.MODEL_BUNDLE_PATH = os.path.join(_tmp, "model_bundle.joblib")

from src import db, features, train

random.seed(42)
np.random.seed(42)

TEAMS = {
    "Premier League": {9825: "Arsenal", 8455: "Chelsea", 8650: "Liverpool",
                       8456: "Man City", 8586: "Spurs", 10261: "Newcastle"},
    "La Liga": {8633: "Real Madrid", 8634: "Barcelona", 9906: "Atletico", 8302: "Sevilla"},
}
PLAYERS_PER_TEAM = 5
POSITIONS = ["FWD", "FWD", "MID", "MID", "DEF"]
SEASON = config.CURRENT_SEASON
start = datetime(2026, 8, 15)

print(f"Building synthetic season in {_tmp} ...")
player_rows, match_rows, team_rows = [], [], []
match_id = 1000
for comp, teams in TEAMS.items():
    ids = list(teams)
    for week in range(10):
        date = (start + timedelta(weeks=week)).strftime("%Y-%m-%d")
        random.shuffle(ids)
        for i in range(0, len(ids) - 1, 2):
            home, away = ids[i], ids[i + 1]
            match_id += 1
            match_rows.append({
                "match_id": str(match_id), "season": SEASON, "competition": comp,
                "league_id": config.COMPETITIONS[comp], "round": str(week + 1), "date": date,
                "kickoff_utc": f"{date}T15:00:00Z", "home_team_id": home, "home_team": teams[home],
                "away_team_id": away, "away_team": teams[away], "home_score": random.randint(0, 3),
                "away_score": random.randint(0, 3), "status": "completed",
                "fixtures_synced_at": datetime.now().isoformat(),
            })
            for team, opp, is_home in [(home, away, 1), (away, home, 0)]:
                base_shots = 2.0 + (team % 3)   # fixed per-team strength so opponent logic has signal
                team_shots = 0
                for p in range(PLAYERS_PER_TEAM):
                    minutes = random.choice([90, 90, 90, 60, 30])
                    shots = int(np.random.poisson(base_shots / PLAYERS_PER_TEAM * 2 * minutes / 90))
                    sot = min(shots, int(np.random.poisson(max(shots * 0.4, 0.1))))
                    team_shots += shots
                    player_rows.append({
                        "match_id": str(match_id), "player_id": team * 100 + p,
                        "player_name": f"{teams[team]} Player {p}", "team_id": team,
                        "team": teams[team], "opponent_id": opp, "opponent": teams[opp],
                        "is_home": is_home, "date": date, "competition": comp, "season": SEASON,
                        "started": int(minutes >= 60), "position": POSITIONS[p], "position_id": 115,
                        "minutes": minutes, "shots": shots, "shots_on_target": sot,
                        "penalty_shots": 0, "goals": 0, "xg": shots * 0.1, "npxg": shots * 0.1,
                        "touches_opp_box": shots * 2.0, "rating": 6.8,
                    })
                team_rows.append({
                    "match_id": str(match_id), "team_id": team, "team": teams[team], "opponent_id": opp,
                    "opponent": teams[opp], "is_home": is_home, "date": date, "competition": comp,
                    "season": SEASON, "shots_for": team_shots, "shots_against": None, "sot_for": None,
                    "sot_against": None, "xg_for": 1.2, "xg_against": 1.1, "possession": 50.0,
                    "coach_id": team * 10, "coach_name": "Coach",
                })

raw_df, matches_df, teams_df = pd.DataFrame(player_rows), pd.DataFrame(match_rows), pd.DataFrame(team_rows)
# shots_against = the opponent's shots_for in the same match
teams_df["shots_against"] = teams_df.merge(
    teams_df[["match_id", "team_id", "shots_for"]].rename(columns={"team_id": "opponent_id", "shots_for": "sa"}),
    on=["match_id", "opponent_id"], how="left")["sa"].values

db.init_db()
db.upsert_matches(matches_df)
for mid, grp in raw_df.groupby("match_id"):
    db.save_match_details(mid, {"players": grp, "teams": teams_df[teams_df.match_id == mid],
                                "unavailable": pd.DataFrame()})
stored, teams_stored = db.load_player_match_stats(), db.load_team_match_stats()
assert len(stored) == len(raw_df), "every synthetic row should round-trip through the DB"
assert len(teams_stored) == len(teams_df)
assert db.matches_needing_details().empty, "all matches should be marked done"
print(f"Stored {len(stored)} player-match rows across {matches_df.match_id.nunique()} matches.")

print("\nBuilding features...")
feat = features.build_features(stored, teams_stored)
rows = features.training_rows(feat)
print(f"Feature table {feat.shape}, {len(rows)} training rows")
assert not rows.empty, "should have starter rows to train on"
for col in features.FEATURE_COLUMNS:
    assert col in feat.columns, f"missing feature column {col}"
    assert rows[col].notna().all(), f"feature {col} has gaps"

print("\nTraining (model comparison + refit)...")
report = train.train_and_save(rows)
print({t: v["chosen"] for t, v in report["targets"].items()})
assert os.path.exists(config.MODEL_BUNDLE_PATH)

print("\nPredicting an upcoming fixture...")
day = start + timedelta(weeks=11)
fixture = {"match_id": "future001", "date": day.strftime("%Y-%m-%d"),
           "kickoff_utc": day.strftime("%Y-%m-%dT15:00:00Z"), "competition": "Premier League",
           "season": SEASON}
sides = [(9825, "Arsenal", 8455, "Chelsea", 1), (8455, "Chelsea", 9825, "Arsenal", 0)]
up_teams = pd.DataFrame([{**fixture, "team_id": t, "team": n, "opponent_id": o, "opponent": on, "is_home": h}
                         for t, n, o, on, h in sides])
up_players = pd.DataFrame([
    {**fixture, "player_id": t * 100 + i, "player_name": f"{n} Player {i}", "team_id": t, "team": n,
     "opponent_id": o, "opponent": on, "is_home": h, "started": 1, "position": POSITIONS[i],
     "lineup_type": "predicted"}
    for t, n, o, on, h in sides for i in range(PLAYERS_PER_TEAM)
])
feat = features.build_features(stored, teams_stored, up_players, up_teams)
future = feat[feat.is_future == 1]
assert len(future) == len(up_players), "every expected starter should get a feature row"
assert future[features.FEATURE_COLUMNS].notna().all().all(), "upcoming rows must have no gaps"

scored = train.predict(future, config.DEFAULT_SHOT_THRESHOLDS, config.DEFAULT_SOT_THRESHOLDS)
scored["generated_at"] = datetime.now().isoformat()
scored["sample_size_overall"] = scored["form_games"].astype(int)
scored["sample_size_vs_opp"] = scored["h2h_player_n"].astype(int)
db.save_predictions(scored[db.PREDICTION_COLS])
assert len(db.latest_predictions()) == len(scored)
print(scored[["player_name", "position", "opponent", "pred_shots", "pred_sot"]].to_string(index=False))

print("\nAll smoke-test assertions passed.")
