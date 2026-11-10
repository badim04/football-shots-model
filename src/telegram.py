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
from datetime import timedelta, timezone

import pandas as pd
import requests

import config

API = "https://api.telegram.org/bot{token}/{method}"
MAX_MESSAGE_CHARS = 3900   # Telegram's limit is 4096; leave room for HTML entities
LOCAL_TZ = timezone(timedelta(hours=config.LOCAL_UTC_OFFSET_HOURS))


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
               "shots>=4": "4+ shots", "sot>=1": "1+ SOT", "sot>=2": "2+ SOT"}


def local_time(kickoff_utc):
    """A kickoff (or a Series of them) in local time."""
    t = pd.to_datetime(kickoff_utc, utc=True, format="ISO8601")
    return t.dt.tz_convert(LOCAL_TZ) if isinstance(t, pd.Series) else t.tz_convert(LOCAL_TZ)


def _split(lines: list, cont_header: str) -> list:
    """Join lines into messages under Telegram's length limit."""
    messages, current = [], []
    for line in lines:
        if current and sum(len(x) + 1 for x in current) + len(line) > MAX_MESSAGE_CHARS:
            messages.append("\n".join(current))
            current = [cont_header]
        current.append(line)
    if current:
        messages.append("\n".join(current))
    return messages


SHORT_LABELS = {"shots>=1": "1+", "shots>=2": "2+", "shots>=3": "3+", "shots>=4": "4+",
                "sot>=1": "1+ SOT", "sot>=2": "2+ SOT"}


def select_block_picks(preds: pd.DataFrame, top_n: int = None, lines: list = None) -> pd.DataFrame:
    """
    The top_n players by expected shots, as one row per (player, line) with
    that line's probability - the form both the post and the results use.
    """
    top_n = top_n or config.TELEGRAM_TOP_N
    lines = lines or config.TELEGRAM_LINES
    top = preds.sort_values("pred_shots", ascending=False).head(top_n).reset_index(drop=True)
    top["rank"] = top.index + 1
    probs = top["prob_json"].apply(json.loads)
    out = [top.assign(line=line, prob=probs.apply(lambda d: d.get(line))) for line in lines]
    picks = pd.concat(out, ignore_index=True)
    return picks[picks.prob.notna()].sort_values(["rank", "line"], kind="stable").reset_index(drop=True)


def _players(picks: pd.DataFrame):
    """(first row, that player's lines) per player, in rank order."""
    order = picks.drop_duplicates(["match_id", "player_id"])
    for r in order.itertuples():
        yield r, picks[(picks.match_id == r.match_id) & (picks.player_id == r.player_id)]


def build_block_post(picks: pd.DataFrame, title: str, n_games: int) -> list:
    """The block's top players, each with every line's probability and fair odds."""
    if picks.empty:
        return []
    lines = [f"⚽ <b>{html.escape(title)}</b>",
             f"<i>Top {picks.player_id.nunique()} players by expected shots · {n_games} games · "
             f"kickoff {config.LOCAL_TZ_LABEL} · % = model probability, @ = fair odds</i>"]
    for i, (r, own) in enumerate(_players(picks), 1):
        ko = local_time(r.kickoff_utc).strftime("%a %H:%M")
        notes = (" ⚠️" if getattr(r, "sample_size_overall", 7) < config.MIN_SAMPLE_SIZE_WARNING else "") + \
                (" (XI est.)" if getattr(r, "lineup_type", None) == "estimated" else "")
        odds = " · ".join(f"{SHORT_LABELS.get(p.line, p.line)} {p.prob:.0%} @{1 / p.prob:.2f}"
                          for p in own.itertuples())
        lines += ["", f"{i}. <b>{html.escape(r.player_name)}</b> ({html.escape(r.team)} v "
                      f"{html.escape(r.opponent)}, {ko}) <i>xShots {r.pred_shots:.1f}</i>{notes}",
                  f"    {odds}"]
    lines += ["", f"💡 Bet a line only when the bookmaker pays at least {config.VALUE_MARGIN:.2f}× "
                  f"the fair odds (e.g. @2.00 → need 2.20+). Below that, skip it however likely it looks.",
              f"<i>{html.escape(config.TELEGRAM_FOOTER, quote=False)}</i>"]
    return _split(lines, f"⚽ <b>{html.escape(title)}</b> (cont.)")


def build_results_post(settled: pd.DataFrame, title: str, season: pd.DataFrame) -> list:
    """
    How a block's posted players did: each player's shots/SOT with ✅/❌ per
    line (➖ = didn't play, void), hits vs expected per line, and the
    running total over every block posted so far.
    """
    lines = [f"📊 <b>{html.escape(title)}</b>", ""]
    for r, own in _players(settled):
        if not r.played:
            lines.append(f"➖ {html.escape(r.player_name)} — didn't play")
            continue
        marks = " ".join(f"{SHORT_LABELS.get(p.line, p.line)} {'✅' if p.hit else '❌'}" for p in own.itertuples())
        lines.append(f"{html.escape(r.player_name)}: {int(r.shots)} shots, {int(r.shots_on_target)} SOT — {marks}")

    def summary(df):
        live = df[df.played]
        return [f"• {LINE_LABELS.get(line, line)}: {int(g.hit.sum())}/{len(g)} hit ({g.hit.mean():.0%}), "
                f"model expected {g.prob.mean():.0%}" for line, g in live.groupby("line", sort=False)]
    lines += ["", "<b>This block</b>"] + summary(settled)
    if season.block_key.nunique() > 1:
        lines += ["", f"<b>All {season.block_key.nunique()} blocks so far</b>"] + summary(season)
    lines += ["", "<i>Hit rate close to the expected % = the probabilities can be trusted.</i>"]
    return _split(lines, f"📊 <b>{html.escape(title)}</b> (cont.)")
