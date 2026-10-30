"""
The daily job: fetch new results → retrain if there's new data → predict
upcoming fixtures with FotMob's expected lineups → publish → Telegram.

    python scripts/run_daily.py                          # everything except Telegram
    python scripts/run_daily.py --telegram --dry-run     # print the Telegram posts, send nothing
    python scripts/run_daily.py --telegram               # post picks for the next 36 hours

Run it once a day (Windows Task Scheduler — see README). Each fixture is
posted to Telegram once, so running it more often never double-posts.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

# emoji in Telegram previews would crash a cp1252 console or redirected log
for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import db, pipeline, train


def _needs_retrain(new_matches: int, force: bool) -> bool:
    if force or new_matches > 0 or not os.path.exists(config.MODEL_BUNDLE_PATH):
        return True
    trained = datetime.fromisoformat(train.load_bundle()["trained_at"])
    return datetime.now(timezone.utc) - trained > timedelta(days=7)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7, help="predict fixtures this many days ahead")
    ap.add_argument("--telegram", action="store_true", help="post picks to Telegram")
    ap.add_argument("--dry-run", action="store_true", help="with --telegram: print instead of sending")
    ap.add_argument("--within-hours", type=int, default=36,
                    help="post fixtures kicking off within this many hours")
    ap.add_argument("--min-prob", type=float, default=config.DEFAULT_MIN_PERCENTAGE,
                    help="probability a line must reach to be posted")
    ap.add_argument("--skip-update", action="store_true", help="don't fetch new results first")
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()

    print(f"=== daily run {datetime.now():%Y-%m-%d %H:%M} ===")
    db.init_db()

    new_matches = 0
    if not args.skip_update:
        update = pipeline.weekly_update()
        new_matches = update["details"]["ok"]
        print(f"[update] {new_matches} new match(es) stored")
        if update["details"]["blocked"]:
            print("[update] FotMob refused requests - continuing with the data we have")

    if _needs_retrain(new_matches, args.force_retrain):
        result = pipeline.weekly_retrain()
        if result["status"] == "ok":
            chosen = {t: v["chosen"] for t, v in result["report"]["targets"].items()}
            print(f"[retrain] done - chosen models {chosen}")
        else:
            print(f"[retrain] {result['status']}")
            sys.exit(1)
    else:
        print("[retrain] no new data - keeping the current model")

    preds = pipeline.generate_weekly_predictions(days_ahead=args.days)
    print(f"[predict] {len(preds)} player predictions"
          + (f" across {preds.match_id.nunique()} fixtures" if not preds.empty else
             f" - no fixtures in scope in the next {args.days} days"))
    if not preds.empty:
        lt = preds.drop_duplicates("match_id").lineup_type.value_counts().to_dict()
        print(f"[predict] lineups used: {lt}")

    for path in pipeline.publish_files(preds):
        print(f"[publish] {os.path.relpath(path, config.BASE_DIR)}")

    if args.telegram:
        result = pipeline.post_to_telegram(preds, args.min_prob, args.within_hours, dry_run=args.dry_run)
        print(f"[telegram] {json.dumps(result)}")


if __name__ == "__main__":
    main()
