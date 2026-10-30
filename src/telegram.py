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


def format_match_block(match: pd.Series, preds: pd.DataFrame, min_prob: float) -> str:
    """One fixture's picks, or '' if no player clears min_prob on any line."""
    picks = []
    for _, r in preds.iterrows():
        lines = _qualifying_lines(r["prob_json"], min_prob)
        if lines:
            best = max(v for _, v in lines)
            picks.append((best, r, lines))
    if not picks:
        return ""
    picks.sort(key=lambda x: -x[0])
    picks = picks[:config.TELEGRAM_MAX_PICKS_PER_MATCH]

    kickoff = pd.to_datetime(match["kickoff_utc"], format="ISO8601").strftime("%H:%M UTC")
    lineup = preds["lineup_type"].iloc[0] if "lineup_type" in preds else None
    lineup_note = {"confirmed": "✅ confirmed XI", "predicted": "predicted XI"}.get(lineup, "XI estimated")
    out = [f"<b>{html.escape(match['home_team'])} vs {html.escape(match['away_team'])}</b>",
           f"<i>{html.escape(match['competition'])} · {kickoff} · {lineup_note}</i>"]
    for _, r, lines in picks:
        line_txt = " · ".join(f"{LINE_LABELS[k]} {v:.0%}" for k, v in lines)
        thin = " ⚠️" if r.get("sample_size_overall", 7) < config.MIN_SAMPLE_SIZE_WARNING else ""
        out.append(f"• {html.escape(r['player_name'])} ({html.escape(r['team'])}) — {line_txt} "
                   f"<i>(xShots {r['pred_shots']:.1f}, xSOT {r['pred_sot']:.1f})</i>{thin}")
    return "\n".join(out)


def build_messages(matches: pd.DataFrame, preds: pd.DataFrame, min_prob: float) -> list:
    """
    [(text, [match_ids in this message])] — matches grouped by day, split
    across messages so none exceeds Telegram's length limit.
    """
    messages, current, current_ids = [], [], []

    def flush():
        if current:
            messages.append(("\n\n".join(current), list(current_ids)))
            current.clear()
            current_ids.clear()

    day_header = None
    for _, m in matches.sort_values("kickoff_utc").iterrows():
        block = format_match_block(m, preds[preds.match_id == m["match_id"]], min_prob)
        if not block:
            continue
        day = pd.to_datetime(m["kickoff_utc"], format="ISO8601").strftime("%a %d %b")
        if day != day_header:
            header = f"⚽ <b>Shots picks — {day}</b>  (≥{min_prob:.0%} model probability)"
            if sum(len(x) for x in current) + len(header) > MAX_MESSAGE_CHARS * 0.7:
                flush()
            current.append(header)
            day_header = day
        if sum(len(x) + 2 for x in current) + len(block) > MAX_MESSAGE_CHARS:
            flush()
            current.append(f"⚽ <b>Shots picks — {day}</b> (cont.)")
        current.append(block)
        current_ids.append(m["match_id"])
    if current:
        current.append(f"<i>{html.escape(config.TELEGRAM_FOOTER, quote=False)}</i>")
    flush()
    return messages


def format_track_record(summary: pd.DataFrame, since: str) -> str:
    """A short 'how did the picks do' message from evaluate.track_record()."""
    if summary.empty:
        return ""
    out = [f"📊 <b>Track record since {html.escape(since)}</b>"]
    for _, r in summary.iterrows():
        out.append(f"• {LINE_LABELS.get(r['line'], r['line'])}: {int(r['hits'])}/{int(r['picks'])} hit "
                   f"({r['hit_rate']:.0%}) vs {r['avg_prob']:.0%} predicted")
    return "\n".join(out)
