"""
CLI entrypoint for the scheduled workflow. Also runnable by hand:
    python scripts/run_predict.py              # fixtures in the next 7 days
    python scripts/run_predict.py --days 14    # e.g. across an international break
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="how many days ahead to predict")
    args = ap.parse_args()

    if not os.path.exists(config.MODEL_SHOTS_PATH):
        print("No trained model yet - run scripts/run_retrain.py first.")
        sys.exit(0)
    preds = pipeline.generate_weekly_predictions(days_ahead=args.days)
    if preds.empty:
        print(f"No predictions generated: no fixtures in the next {args.days} days in scope "
              "(international break?), or no history yet for the teams that are playing.")
    else:
        print(f"Generated {len(preds)} prediction rows.")
        print(preds.sort_values("pred_shots", ascending=False).head(15)[
            ["date", "player_name", "team", "opponent", "pred_shots", "pred_sot",
             "sample_size_overall"]].to_string(index=False))
