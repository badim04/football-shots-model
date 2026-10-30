# Football Shots & Shots-on-Target Model

> **Case study:** a plain-English walkthrough of how this works, step by step, with the key code: https://abdulakeem-adebayo.vercel.app/#football

Predicts, per player, expected **shots** and **shots on target** for the
next match, plus the probability of each betting line (1+, 2+, 3+ shots;
1+, 2+ SOT). It covers the Premier League, La Liga, Bundesliga, Serie A,
Ligue 1, Champions League, Europa League and Conference League. It
predicts the **2026-27 season**, learning from everything back to
**2023-24**, and posts the strongest picks to a **Telegram channel**.

Data comes from FotMob's public match data, not a paid API.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/backfill.py
python scripts/run_daily.py --telegram --dry-run
python scripts/run_daily.py --telegram
streamlit run app.py
```

- **`backfill.py`** is a one-time history download, about 7 hours. It can
  be stopped and resumed.
- **`run_daily.py --telegram --dry-run`** fetches new results, retrains,
  predicts, and *prints* the Telegram posts.
- **`run_daily.py --telegram`** does the same and actually posts.
- **`streamlit run app.py`** opens the app.

---

## What the model looks at

Footballers are people, and they aren't equally predictable all the time.
Their output depends on how they've been playing lately, who they're
playing, and the circumstances. The features are built around those
three things. Every one is computed only from matches **before** the one
being predicted.

| Group | What's in it | Why |
|---|---|---|
| **Recent form** | Shots, SOT, xG and touches in the box **per 90 minutes**, weighted toward the last few games (half-life 3 games: last game 100%, three games ago 50%, six ago 25%). Recent minutes and starts are tracked separately. | The last 4–7 games say most about the next one. Per-90 means being subbed at 60' isn't mistaken for bad form. |
| **Normal level** | A slower, longer-run rate, pulled toward the position's average while history is thin. Also the player's share of team shots and penalty duty. | A fallback when form rests on 1–2 games, and a reference point: form *relative to normal* is a feature in itself. |
| **Head-to-head** | Team level: how many shots this team takes against this opponent compared with its usual level, across seasons, weighted toward recent meetings. Player level: the same for the player. Number of meetings and how long ago. Derby flag. | Some teams can't cope with a particular opponent's setup (Madrid vs Atlético), and derbies play differently. Pulled toward "no effect" until there are a few meetings, so one odd game doesn't dominate. |
| **Team context** | Team attacking form, the opponent's shots conceded relative to its competition, opponent possession and attack (game state), home/away, European game. | A striker facing a deep block, or chasing a game, shoots differently. |
| **Circumstances** | Rest days for both teams, a European game in the last 4 days, days since the player last played, games missed (back from injury), games under the current coach. | Fatigue, rust and a new manager all change output. |

**Who is predicted.** FotMob publishes a **predicted XI** and injury list
days before each match, and the **confirmed XI** about an hour before
kickoff. Predictions use those. When neither is out yet, they fall back to
players who started at least 3 of the team's last 5 matches. Every
prediction records which of the three it used. The model learns from
starters only, so it answers "how many shots, given he starts".

## How it decides what works

`scripts/evaluate.py` settles questions with held-out data instead of
opinion. It trains on the past, tests on the most recent 4 weeks, and
reports:

- **Models:** a simple baseline (form × usual minutes × opponent), Poisson
  regression (GLM) and XGBoost. Each retrain picks whichever predicts the
  holdout weeks best, so the baseline has to be beaten for the more complex
  models to be used.
- **Feature groups:** how much worse the model gets when each group (form,
  H2H, …) is removed. A big number means the group carries real
  information.
- **Form half-life:** `--halflives 1 2 3 5 8` compares how "recent" form
  should be.

**Probabilities.** P(shots ≥ k) comes from a count distribution around
the predicted average. Shots vary more than a Poisson distribution
allows, so a negative binomial is used when it fits the holdout better.
This mostly affects the 2+ and 3+ lines.

**Track record.** Every prediction is stored with a timestamp. After the
match, the last pre-kickoff prediction is compared with what happened.
The app's *Track record* tab shows, per line, how many picks hit against
how many the model expected. If "70%" picks land about 70% of the time,
the model is calibrated.

First results (October 2026, current season only, 2,617 holdout starters):
the GLM beat the baseline clearly, with shots deviance 1.32 vs 1.57.
Picks rated about 75% for 1+ shots landed 74% of the time, and picks
rated 85% landed 88%. Head-to-head can only show its value once past
seasons are in. Re-run `scripts/evaluate.py` after the backfill.

---

## Telegram

1. In Telegram, message **@BotFather**, send `/newbot`, and copy the token.
2. Add the bot to your channel as an **administrator** with "Post messages".
3. Create a file called `.env` in the project folder. It's git-ignored, so
   it never gets committed:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC-your-token
   TELEGRAM_CHAT_ID=@your_channel_username
   ```
   For a private channel, use its numeric id (`-100…`) instead.
4. Preview what will be posted, then post for real:
   ```bash
   python scripts/run_daily.py --telegram --dry-run
   python scripts/run_daily.py --telegram
   ```

Each day gets **one post**: the **top 20 players** across every game
kicking off in the next 24 hours, best first. A player only makes the list
if one of their lines is at least **80%** likely. The post shows the
player, the game and kickoff, the lines that clear 80% (at most 2), and
xShots. ⚠️ means thin history, and "XI est." means FotMob hasn't
published a lineup yet. The job posts at most once a day, and a game
already covered is never posted again. Change the defaults with `--top`,
`--min-prob` and `--within-hours`, or with `TELEGRAM_TOP_N` and
`TELEGRAM_MIN_PROB` in `config.py`.

## Running it every day (Windows Task Scheduler)

This creates a task that runs the daily job at 09:00:

```bash
schtasks /Create /TN "Football shots daily" /SC DAILY /ST 09:00 /TR "cmd /c cd /d C:\Users\user\Documents\GitHub\football-shots-model && python scripts\run_daily.py --telegram >> logs\daily.log 2>&1"
```

The PC has to be on and online. If it was off at 09:00, the task runs as
soon as it can (enable "Run task as soon as possible after a scheduled
start is missed" in the task's Settings tab). Output goes to
`logs/daily.log`.

## The app

```bash
streamlit run app.py
```

- **Predictions:** every expected starter in the next 7 days, with
  per-line probabilities and filters.
- **Track record:** hit rates, calibration, how often predicted starters
  actually played, and the latest model comparison.
- **Admin:** fetch, retrain and predict buttons. Only shown when running
  locally with the database.

To deploy on Streamlit Community Cloud, point it at `app.py`. The full
database stays on your PC, about 70 MB with history, and isn't in git.
The deployed app reads the small files in `data/published/`, which
`run_daily.py` rewrites every run. Commit and push those files to update
the deployed app.

---

## Data

### Why FotMob

- **FBref** lost its Opta licence in January 2026, and per-player shots
  went with it. It also serves a Cloudflare challenge that a normal client
  can't pass.
- **Paid APIs** charge roughly $20–100+/month for this depth of data.
- **FotMob's** match JSON has minutes, shots, SOT and xG per player, plus
  starters, injuries, coach, formation and possession, with stable IDs.

The endpoints are **unofficial and undocumented**. They can change or
start refusing requests. All parsing lives in `scraper.parse_match_details`,
and every raw response is cached, so a parser fix only needs
`backfill.py --reparse`.

### Politeness

- One request per competition-season for fixtures, and one per match.
- A finished match is fetched once, ever (raw JSON gzipped under
  `data/raw_cache/`).
- A 2–4 s randomized delay, and backoff on 429/5xx.
- A **403 stops the run**, with progress saved, instead of retrying or
  trying to get around the block.
- A match counts as done only once its stats are stored. Failures are
  retried, up to 3 attempts.

### Backfill

```bash
python scripts/backfill.py --dry-run
python scripts/backfill.py
python scripts/backfill.py --seasons 2025/2026
python scripts/backfill.py --reparse
python scripts/backfill.py --retry-given-up
```

`--dry-run` shows what's left and how long it'll take. `--seasons`
limits the run to some seasons. `--reparse` rebuilds the tables from the
cache with no network. `--retry-given-up` retries matches that failed 3
times.

---

## Project layout

```
app.py                     Streamlit app
config.py                  every tunable setting (form half-life, derbies, delays, ...)
src/scraper.py             FotMob client + parsers (results, predicted/confirmed lineups)
src/db.py                  SQLite schema + helpers (ID-keyed; additive migrations)
src/features.py            feature groups above; one builder for training AND prediction
src/train.py               baseline / GLM / XGBoost, model selection, Poisson vs neg. binomial
src/evaluate.py            track record + calibration of published predictions
src/telegram.py            Telegram formatting + sending
src/pipeline.py            orchestration: update, retrain, lineups, predict, publish, post
scripts/run_daily.py       the daily job
scripts/backfill.py        one-time history download
scripts/evaluate.py        model comparison, feature-group ablation, form half-life
scripts/run_weekly_scrape.py / run_retrain.py / run_predict.py   individual steps
scripts/smoke_test.py      end-to-end check on synthetic data (own temp DB)
notebooks/explore.ipynb    player logs, H2H tables, coverage
data/published/            small files the deployed app reads (committed)
```

## Failure modes worth knowing about

- **FotMob changes its JSON or blocks requests.** Matches come back
  `no_stats`/`failed`, or the run stops with a 403. Nothing already stored
  is lost.
- **The predicted XI is wrong.** The Track record tab shows how often
  predicted starters actually played. Posts within an hour of kickoff use
  the confirmed XI, but each fixture is posted only once, at the first
  daily run inside the 36-hour window.
- **Thin history.** New signings from outside these 8 competitions, and
  promoted teams, start from position averages. Predictions show
  "recent games" and a ⚠️ in Telegram.
- **Domestic cups aren't in the data.** Rest days don't see an FA Cup or
  Copa del Rey game in between.
- **International breaks.** No fixtures means no predictions. Use
  `run_predict.py --days 14` to look further ahead.
