"""
The scheduled job: fetch new results → retrain if there's new data →
predict upcoming fixtures with FotMob's expected lineups → publish →
Telegram block posts and results.

    python scripts/run_daily.py --auto                   # what the scheduler runs
    python scripts/run_daily.py --telegram --dry-run     # preview the current/next block + any results
    python scripts/run_daily.py                          # update, retrain, predict 7 days; no Telegram

--auto is meant to run every couple of hours. It does nothing unless a
Telegram post is due (times in Lagos time, see POST_BLOCKS in config.py),
so missed runs - PC off, no power or internet - are simply caught up by
the next one:
  * a block post: fixtures are grouped Tue-Thu, Fri-Sat, Sun-Mon; each
    block is posted once, from 10:00 the day before it starts;
  * a results post: once all of a posted block's games are finished and
    their stats are in.
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
    ap.add_argument("--auto", action="store_true",
                    help="scheduler mode: only run when a Telegram post is due, then post it")
    ap.add_argument("--telegram", action="store_true", help="post whatever is due to Telegram")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --telegram: print the current/next block and pending results, send nothing")
    ap.add_argument("--days", type=int, default=7, help="predict fixtures this many days ahead")
    ap.add_argument("--skip-update", action="store_true", help="don't fetch new results first")
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()
    telegram_on = args.auto or args.telegram

    db.init_db()
    now = pipeline.local_now()
    blocks = pipeline.due_blocks(now, preview=args.dry_run) if telegram_on else []
    results = pipeline.due_results(now) if telegram_on else []
    if args.auto and not blocks and not results:
        return   # nothing due - stay quiet so the log only shows real runs

    print(f"=== run {now:%a %Y-%m-%d %H:%M} {config.LOCAL_TZ_LABEL} ===")
    if blocks:
        print(f"[due] block posts: {[b['key'] for b in blocks]}")
    if results:
        print(f"[due] results: {results}")

    new_matches = 0
    if not args.skip_update:
        update = pipeline.weekly_update()
        new_matches = update["details"]["ok"]
        print(f"[update] {new_matches} new match(es) stored")
        if update["details"]["blocked"]:
            print("[update] FotMob refused requests - continuing with the data we have")

    for key in results:
        print(f"[results] {json.dumps(pipeline.post_results(key, dry_run=args.dry_run))}")

    if args.auto and not blocks:
        return   # results-only run: no need to retrain or predict

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

    days = args.days
    if blocks:
        days = max(days, max((b["end"] - now.date()).days + 1 for b in blocks))
    preds = pipeline.generate_weekly_predictions(days_ahead=days)
    print(f"[predict] {len(preds)} player predictions"
          + (f" across {preds.match_id.nunique()} fixtures" if not preds.empty else
             f" - no fixtures in scope in the next {days} days"))
    if not preds.empty:
        lt = preds.drop_duplicates("match_id").lineup_type.value_counts().to_dict()
        print(f"[predict] lineups used: {lt}")

    for path in pipeline.publish_files(preds):
        print(f"[publish] {os.path.relpath(path, config.BASE_DIR)}")

    for b in blocks:
        print(f"[telegram] {json.dumps(pipeline.post_block(preds, b, dry_run=args.dry_run))}")


if __name__ == "__main__":
    main()
