"""
Dev tool: a sanity check that storage, feature engineering, training,
drift and prediction all run end-to-end, on synthetic data. Uses its own
throwaway database and model directory, so it never touches your real
backfilled data in data/matches.db or models/.
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

config.MODEL_SHOTS_PATH = os.path.join(_tmp, "model_shots.joblib")
config.MODEL_SOT_PATH = os.path.join(_tmp, "model_sot.joblib")
config.FEATURE_LIST_PATH = os.path.join(_tmp, "feature_list.json")

from src import db, features, train, drift

random.seed(42)
np.random.seed(42)

TEAMS = {
    "Premier League": {9825: "Arsenal", 8455: "Chelsea", 8650: "Liverpool",
                       8456: "Man City", 8586: "Spurs", 10261: "Newcastle"},
    "La Liga": {8633: "Real Madrid", 8634: "Barcelona", 9906: "Atletico", 8302: "Sevilla"},
}
PLAYERS_PER_TEAM = 5
SEASON = config.CURRENT_SEASON
start = datetime(2026, 8, 15)

print(f"Building synthetic season in {_tmp} ...")
player_rows, match_rows = [], []
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
                for p in range(PLAYERS_PER_TEAM):
                    minutes = random.choice([90, 90, 90, 60, 30])
                    shots = int(np.random.poisson(base_shots / PLAYERS_PER_TEAM * 2 * minutes / 90))
                    sot = min(shots, int(np.random.poisson(max(shots * 0.4, 0.1))))
                    player_rows.append({
                        "match_id": str(match_id), "player_id": team * 100 + p,
                        "player_name": f"{teams[team]} Player {p}", "team_id": team,
                        "team": teams[team], "opponent_id": opp, "opponent": teams[opp],
                        "is_home": is_home, "date": date, "competition": comp, "season": SEASON,
                        "started": int(minutes >= 60), "position": "FWD", "position_id": 115,
                        "minutes": minutes, "shots": shots, "shots_on_target": sot,
                        "penalty_shots": 0, "goals": 0, "xg": shots * 0.1, "npxg": shots * 0.1,
                        "touches_opp_box": shots * 2.0, "rating": 6.8,
                    })

raw_df = pd.DataFrame(player_rows)
matches_df = pd.DataFrame(match_rows)

db.init_db()
db.upsert_matches(matches_df)
for mid, grp in raw_df.groupby("match_id"):
    db.save_match_details(mid, {"players": grp, "teams": pd.DataFrame(), "unavailable": pd.DataFrame()})
stored = db.load_player_match_stats()
assert len(stored) == len(raw_df), "every synthetic row should round-trip through the DB"
assert db.matches_needing_details().empty, "all matches should be marked done"
print(f"Stored {len(stored)} player-match rows across {matches_df.match_id.nunique()} matches.")

print("\nBuilding features...")
feat = features.build_feature_table(stored, config.RECENT_N_OVERALL, config.RECENT_N_OPPONENT)
print(f"Feature table shape: {feat.shape}")
assert not feat.empty, "feature table should not be empty"
for col in features.FEATURE_COLUMNS:
    assert col in feat.columns, f"missing feature column {col}"

print("\nTraining...")
metrics = train.train_and_save(feat)
print(metrics)
assert os.path.exists(config.MODEL_SHOTS_PATH) and os.path.exists(config.MODEL_SOT_PATH)

print("\nRunning drift check...")
print(drift.run_drift_check(feat))

print("\nBuilding upcoming fixture rows + predicting...")
upcoming = pd.DataFrame([{
    "match_id": "future001", "date": (start + timedelta(weeks=11)).strftime("%Y-%m-%d"),
    "competition": "Premier League", "season": SEASON,
    "home_team_id": 9825, "home_team": "Arsenal", "away_team_id": 8455, "away_team": "Chelsea",
}])
pred_rows = features.build_upcoming_feature_rows(stored, upcoming, config.RECENT_N_OVERALL,
                                                 config.RECENT_N_OPPONENT)
print(f"Upcoming feature rows: {len(pred_rows)}")
assert not pred_rows.empty, "should generate at least one upcoming prediction row"

scored = train.predict(pred_rows, config.DEFAULT_SHOT_THRESHOLDS, config.DEFAULT_SOT_THRESHOLDS,
                       config.DEFAULT_MIN_PERCENTAGE)
scored["generated_at"] = datetime.now().isoformat()
db.save_predictions(scored[db.PREDICTION_COLS])
assert len(db.latest_predictions()) == len(scored)
print(scored[["player_name", "team", "opponent", "pred_shots", "pred_sot", "prob_json"]].head(10).to_string())

print("\nAll smoke-test assertions passed.")
