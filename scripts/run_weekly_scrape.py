"""
CLI entrypoint for the scheduled workflow. Also runnable by hand:
    python scripts/run_weekly_scrape.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline

if __name__ == "__main__":
    summary = pipeline.weekly_update()
    details = summary["details"]
    print(json.dumps({k: v for k, v in details.items() if k != "failures"}, indent=2))
    if details["failures"]:
        print(f"WARNING: {len(details['failures'])} match(es) failed (retried next run):")
        for match_id, err in details["failures"]:
            print(f"  {match_id}: {err}")
    if details["blocked"]:
        print("FotMob refused requests (403) - stopped early; the next run continues from here.")
        sys.exit(2)
