"""
Posting predictions to a Telegram channel via the Bot API.

Setup (once):
  1. In Telegram, message @BotFather → /newbot → copy the bot token.
  2. Add the bot to your channel as an administrator (it needs "Post messages").
  3. Create a file named `.env` in the project root (it's git-ignored) with:
         TELEGRAM_BOT_TOKEN=123456:ABC...
         TELEGRAM_CHAT_ID=@your_channel_username      (or the -100... numeric id)
  4. python scripts/run_daily.py --telegram --dry-run   → prints exactly what would be posted.

The token is only ever read from the environment / `.env` — it is never
written to the database, logs, or committed files.
"""

import html
import json
import os
from datetime import datetime

import pandas as pd
import requests

import config

API = "https://api.telegram.org/bot{token}/{method}"
MAX_MESSAGE_CHARS = 3900   # Telegram's limit is 4096; leave room for HTML entities


def _load_dotenv():
    path = os.path.join(config.BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                # the project's .env wins over machine-wide variables (another
                # project's bot may be set globally)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


def credentials():
    """(token, chat_id), or (None, None) if not configured."""
    _load_dotenv()
    return os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


def send_message(text: str, token: str, chat_id: str) -> int:
    resp = requests.post(API.format(token=token, method="sendMessage"), json={
        "chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True,
    }, timeout=30)
    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if not data.get("ok"):
        # Never include the token in the error.
        raise RuntimeError(f"Telegram sendMessage failed ({resp.status_code}): "
                           f"{data.get('description', resp.text[:200])}")
    return data["result"]["message_id"]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

LINE_LABELS = {"shots>=1": "1+ shots", "shots>=2": "2+ shots", "shots>=3": "3+ shots",
               "sot>=1": "1+ SOT", "sot>=2": "2+ SOT"}


def _qualifying_lines(prob_json: str, min_prob: float) -> list:
    probs = json.loads(prob_json)
    hits = [(k, v) for k, v in probs.items() if v >= min_prob and k in LINE_LABELS]
    # The most demanding line a player clears is the most informative one;
    # show at most two per player.
    hits.sort(key=lambda kv: (kv[0].startswith("shots"), -int(kv[0].split(">=")[1])))
    return hits[:2]


def _pick_line(r: pd.Series, lines: list) -> str:
    line_txt = " · ".join(f"{LINE_LABELS[k]} {v:.0%}" for k, v in lines)
    kickoff = pd.to_datetime(r["kickoff_utc"], format="ISO8601").strftime("%H:%M")
    notes = (" ⚠️" if r.get("sample_size_overall", 7) < config.MIN_SAMPLE_SIZE_WARNING else "") +             (" (XI est.)" if r.get("lineup_type") == "estimated" else "")
    return (f"<b>{html.escape(r['player_name'])}</b> ({html.escape(r['team'])} v "
            f"{html.escape(r['opponent'])}, {kickoff}) — {line_txt} "
            f"<i>(xShots {r['pred_shots']:.1f})</i>{notes}")


def build_digest(preds: pd.DataFrame, min_prob: float, top_n: int, title: str) -> list:
    """
    One ranked list of the most confident players across all the given
    fixtures: everyone with a line at >= min_prob, best first, capped at
    top_n. Returns [(text, [match_ids covered])], split to fit Telegram's
    length limit; [] if nobody qualifies.
    """
    picks = []
    for _, r in preds.iterrows():
        lines = _qualifying_lines(r["prob_json"], min_prob)
        if lines:
            picks.append((max(v for _, v in lines), r, lines))
    if not picks:
        return []
    picks.sort(key=lambda x: -x[0])
    picks = picks[:top_n]

    match_ids = sorted(preds.match_id.unique())
    header = (f"⚽ <b>{html.escape(title)}</b>\n"
              f"<i>Top {len(picks)} players at ≥{min_prob:.0%} model probability · "
              f"{len(match_ids)} games · kickoff times UTC</i>")
    footer = f"<i>{html.escape(config.TELEGRAM_FOOTER, quote=False)}</i>"
    messages, current = [], [header]
    for i, (_, r, lines) in enumerate(picks, 1):
        row = f"{i}. {_pick_line(r, lines)}"
        if sum(len(x) + 1 for x in current) + len(row) > MAX_MESSAGE_CHARS:
            messages.append("\n".join(current))
            current = [f"⚽ <b>{html.escape(title)}</b> (cont.)"]
        current.append(row)
    current += ["", footer]
    messages.append("\n".join(current))
    return [(m, match_ids) for m in messages]


def format_track_record(summary: pd.DataFrame, since: str) -> str:
    """A short 'how did the picks do' message from evaluate.track_record()."""
    if summary.empty:
        return ""
    out = [f"📊 <b>Track record since {html.escape(since)}</b>"]
    for _, r in summary.iterrows():
        out.append(f"• {LINE_LABELS.get(r['line'], r['line'])}: {int(r['hits'])}/{int(r['picks'])} hit "
                   f"({r['hit_rate']:.0%}) vs {r['avg_prob']:.0%} predicted")
    return "\n".join(out)
