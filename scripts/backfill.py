"""
One-time (and resumable) multi-season backfill. Run it on your own PC:

    python scripts/backfill.py --dry-run      # fixture lists only, then show what's left to fetch
    python scripts/backfill.py                # fetch everything in config.BACKFILL_SEASONS
    python scripts/backfill.py --seasons 2025/2026 2026/2027 --limit 200
    python scripts/backfill.py --reparse      # rebuild tables from the local cache, no network

Ctrl+C at any time is safe: each match is committed as it's fetched, and a
re-run carries on from where it stopped. Raw responses are cached under
data/raw_cache/, so nothing already fetched is ever requested again.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import db, pipeline


def _duration(seconds: float) -> str:
    return f"{seconds / 3600:.1f} hours" if seconds >= 3600 else f"{seconds / 60:.0f} minutes"


def _print_progress():
    prog = db.scrape_progress()
    if prog.empty:
        print("No completed matches known yet.")
        return
    table = prog.pivot_table(index=["season", "competition"], columns="details_status",
                             values="n", fill_value=0, aggfunc="sum")
    print(table.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=config.BACKFILL_SEASONS,
                    help=f"FotMob season labels (default: {' '.join(config.BACKFILL_SEASONS)})")
    ap.add_argument("--competitions", nargs="+", choices=list(config.COMPETITIONS),
                    help="limit to these competitions (default: all)")
    ap.add_argument("--limit", type=int, help="fetch at most this many matches this run")
    ap.add_argument("--dry-run", action="store_true",
                    help="sync fixture lists only and report how many matches are left to fetch")
    ap.add_argument("--reparse", action="store_true",
                    help="rebuild player/team tables from the raw cache (no network)")
    ap.add_argument("--retry-given-up", action="store_true",
                    help=f"retry matches that already failed {config.MAX_DETAIL_ATTEMPTS} times")
    args = ap.parse_args()

    db.init_db()

    if args.reparse:
        pipeline.reparse_from_cache(args.seasons)
        _print_progress()
        return

    print(f"Syncing fixture lists for {', '.join(args.seasons)}...")
    pipeline.sync_fixtures(args.seasons, args.competitions)

    if args.retry_given_up:
        db.reset_detail_attempts(args.seasons)

    todo = db.matches_needing_details(args.seasons, args.competitions)
    n = min(len(todo), args.limit) if args.limit else len(todo)
    avg_delay = (config.MIN_DELAY_SECONDS + config.MAX_DELAY_SECONDS) / 2 + 0.4
    print(f"\n{len(todo)} completed matches still need stats"
          + (f" (fetching {n} this run)" if args.limit else "")
          + f" - roughly {_duration(n * avg_delay)} at the configured delay "
          "(already-cached matches are instant).")
    if args.dry_run:
        _print_progress()
        return

    summary = pipeline.scrape_pending_details(args.seasons, args.competitions, limit=args.limit)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if summary["failures"]:
        print(f"{len(summary['failures'])} match(es) failed (they'll be retried next run):")
        for match_id, err in summary["failures"][:20]:
            print(f"  {match_id}: {err}")
    print()
    _print_progress()
    if summary["blocked"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
