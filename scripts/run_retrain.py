"""
CLI entrypoint for the scheduled workflow. Also runnable by hand:
    python scripts/run_retrain.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline

if __name__ == "__main__":
    result = pipeline.weekly_retrain()
    print(json.dumps(result, indent=2, default=str))
    if result["status"] == "no_data":
        print("No data in the database — nothing to train on yet.")
        sys.exit(0)
