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
import re
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


def _split(chunks: list, cont_header: str) -> list:
    """
    Join chunks into messages under Telegram's length limit. A chunk (e.g.
    one day's collapsible list) is never cut in half, so tags stay balanced.
    """
    def visible(s):   # Telegram's limit counts the text after tags are parsed
        return len(html.unescape(re.sub(r"<[^>]+>", "", s)))

    messages, current = [], []
    for chunk in chunks:
        if current and sum(visible(x) + 1 for x in current) + visible(chunk) > MAX_MESSAGE_CHARS:
            messages.append("\n".join(current))
            current = [cont_header]
        current.append(chunk)
    if current:
        messages.append("\n".join(current))
    return messages


def _collapsed(lines: list) -> str:
    """A tap-to-expand quote block: the post stays short until it's opened."""
    return "<blockquote expandable>" + "\n".join(lines) + "</blockquote>"


# Fixed-width labels so the probability/odds columns line up in <code>
SHORT_LABELS = {"shots>=1": "1+ ", "shots>=2": "2+ ", "shots>=3": "3+ ", "shots>=4": "4+ ",
                "sot>=1": "SOT", "sot>=2": "2SOT"}
RESULT_ORDER_NOTE = "ticks = 2+ · 3+ · 4+ shots · 1+ SOT"


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
    """(first row, that player's lines) per player, in the order given."""
    order = picks.drop_duplicates(["match_id", "player_id"])
    for r in order.itertuples():
        yield r, picks[(picks.match_id == r.match_id) & (picks.player_id == r.player_id)]


def _cell(p) -> str:
    return f"{SHORT_LABELS.get(p.line, p.line)} {p.prob:>4.0%} @{1 / p.prob:<4.2f}"


def _block_dates(picks: pd.DataFrame) -> str:
    days = sorted(local_time(picks.kickoff_utc).dt.date.unique())
    first, last = days[0], days[-1]
    return f"{first:%a %d %b}" + (f" – {last:%a %d %b}" if last != first else "")


def build_block_post(picks: pd.DataFrame, title: str, n_games: int) -> list:
    """
    At a glance: the block, then the 5 best 3+ shots lines. Tap a day to
    see every player in it, grouped by game, with each line's probability
    and fair odds.
    """
    if picks.empty:
        return []
    picks = picks.assign(ko=local_time(picks.kickoff_utc))
    head = [f"⚽ <b>SHOTS PICKS · {_block_dates(picks)}</b>",
            f"<i>{n_games} games · top {picks.player_id.nunique()} shooters · times {config.LOCAL_TZ_LABEL}</i>"]

    star_line = "shots>=3" if "shots>=3" in set(picks.line) else picks.line.iloc[0]
    star = picks[picks.line == star_line].sort_values("prob", ascending=False).head(5)
    head += ["", f"⭐ <b>Best {LINE_LABELS.get(star_line, star_line)}</b>"]
    for r in star.itertuples():
        head.append(f"<code>{r.prob:>4.0%} @{1 / r.prob:.2f}</code>  <b>{html.escape(r.player_name)}</b> "
                    f"· {r.ko:%a %H:%M}")

    chunks = ["\n".join(head)]
    by_day = picks.sort_values(["ko", "match_id", "rank"])
    for day, d in by_day.groupby(by_day.ko.dt.date, sort=True):
        body = []
        for _, g in d.groupby(["ko", "match_id"], sort=True):
            first = g.iloc[0]
            label = first.get("match_label") or f"{first.team} v {first.opponent}"
            body += ["", f"🕒 <b>{first.ko:%H:%M}  {html.escape(label)}</b>"]
            for r, own in _players(g):
                cells = [_cell(p) for p in own.itertuples()]
                rows = [" │ ".join(cells[i:i + 2]) for i in range(0, len(cells), 2)]
                notes = (" ⚠️" if getattr(r, "sample_size_overall", 7) < config.MIN_SAMPLE_SIZE_WARNING else "") + \
                        (" · XI est." if getattr(r, "lineup_type", None) == "estimated" else "")
                body.append(f"👤 <b>{html.escape(r.player_name)}</b> ({html.escape(r.team)}) "
                            f"· xShots {r.pred_shots:.1f}{notes}")
                body += [f"<code>{row}</code>" for row in rows]
        n = d.player_id.nunique()
        chunks.append(f"\n📅 <b>{day:%A %d %b}</b> — {n} player{'s' * (n != 1)} · tap to open\n"
                      + _collapsed(body[1:]))

    chunks.append("\n" + _collapsed([
        "<b>How to read it</b>",
        "% = model probability · @ = fair odds (1 ÷ probability).",
        f"Bet a line only if the bookmaker pays at least {config.VALUE_MARGIN:.2f}× the fair odds "
        "(fair @2.00 → need 2.20+). Lower than that, skip it however likely it looks.",
        "⚠️ = little recent history · XI est. = no FotMob lineup yet.",
        f"<i>{html.escape(config.TELEGRAM_FOOTER, quote=False)}</i>"]))
    return _split(chunks, f"⚽ <b>SHOTS PICKS · {_block_dates(picks)}</b> (cont.)")


def _scoreboard(df: pd.DataFrame) -> list:
    """One aligned row per line: hits/picks, hit rate, model expectation."""
    rows = []
    for line, g in df[df.played].groupby("line", sort=False):
        verdict = "🟢" if g.hit.mean() >= g.prob.mean() - 0.02 else ("🟡" if g.hit.mean() >= g.prob.mean() - 0.10 else "🔴")
        rows.append(f"{verdict} <code>{LINE_LABELS.get(line, line):<9}{int(g.hit.sum()):>2}/{len(g):<3}"
                    f"{g.hit.mean():>4.0%}  exp {g.prob.mean():>3.0%}</code>")
    return rows


def build_results_post(settled: pd.DataFrame, title: str, season: pd.DataFrame) -> list:
    """
    At a glance: per-line scoreboard (🟢 at or above what the model
    expected, 🟡 a bit below, 🔴 well below) and the block's best shooter.
    Tap to see every player and the running totals.
    """
    if settled.empty:
        return []
    head = [f"📊 <b>RESULTS · {_block_dates(settled)}</b>", ""] + _scoreboard(settled)
    live = settled[settled.played].drop_duplicates(["match_id", "player_id"])
    if not live.empty:
        best = live.sort_values("shots", ascending=False).iloc[0]
        head += ["", f"🏆 Top shooter: <b>{html.escape(best.player_name)}</b> — "
                     f"{int(best.shots)} shots, {int(best.shots_on_target)} on target"]

    body = [f"<i>{RESULT_ORDER_NOTE}</i>"]
    ordered = settled.assign(_s=settled.shots.fillna(-1)).sort_values("_s", ascending=False, kind="stable")
    for r, own in _players(ordered):
        if not r.played:
            body.append(f"➖➖➖➖ {html.escape(r.player_name)} — didn't play (void)")
            continue
        marks = "".join("✅" if p.hit else "❌" for p in own.itertuples())
        body.append(f"{marks} <b>{html.escape(r.player_name)}</b> — {int(r.shots)} sh · {int(r.shots_on_target)} SOT")
    chunks = ["\n".join(head), "\n👥 <b>Every player</b> · tap to open\n" + _collapsed(body)]
    if season.block_key.nunique() > 1:
        chunks.append(f"\n📈 <b>All {season.block_key.nunique()} blocks so far</b> · tap to open\n"
                      + _collapsed(_scoreboard(season)))
    chunks.append("\n<i>🟢 hit rate at or above the model's % · 🟡 a little below · 🔴 well below</i>")
    return _split(chunks, f"📊 <b>RESULTS · {_block_dates(settled)}</b> (cont.)")
