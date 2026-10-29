# Football Shots & Shots-on-Target Model

> **Case study:** a plain-English walkthrough of how this works, step by step, with the key code: https://abdulakeem-adebayo.vercel.app/#football

Predicts, per player, expected **shots** and **shots on target** for their
next match, plus the probability of hitting configurable thresholds
(≥1, ≥2, ≥3 shots; ≥1, ≥2 SOT). Covers the Premier League, La Liga,
Bundesliga, Serie A, Ligue 1, Champions League, Europa League, and
Conference League. Predictions are for the **2026-27 season**; history
goes back to **2023-24**, which is what head-to-head records and player
baselines are built from.

Data comes from FotMob's public match data, not a paid API — see
*Methodology* below for why, and what that trades off.

---

## Contents

```
football-shots-model/
├── app.py                      # Streamlit app (the web application)
├── config.py                   # every tunable default in one place
├── requirements.txt
├── src/
│   ├── scraper.py              # FotMob client + parser (polite, cached, retried)
│   ├── db.py                   # SQLite schema + read/write helpers
│   ├── features.py             # rolling form, opponent defense rating
│   ├── train.py                # XGBoost training + Poisson thresholding
│   ├── drift.py                # PSI-based drift detection
│   └── pipeline.py             # orchestration — what app.py, scripts & Actions call
├── scripts/
│   ├── backfill.py             # one-time, resumable multi-season backfill (run on your PC)
│   ├── run_weekly_scrape.py    # CLI entrypoint for GitHub Actions
│   ├── run_retrain.py          # CLI entrypoint for GitHub Actions
│   ├── run_predict.py          # CLI entrypoint for GitHub Actions
│   └── smoke_test.py           # synthetic-data end-to-end check (own temp DB)
├── notebooks/
│   └── explore.ipynb           # local workbench: player logs, H2H, coverage
├── .github/workflows/
│   └── weekly_retrain.yml      # the weekly automation
├── data/matches.db             # SQLite database (created by the backfill)
├── data/raw_cache/             # gzipped raw FotMob JSON (git-ignored)
└── models/                     # saved model files (created on first retrain)
```

---

## Methodology

### 1. Why FotMob

- **Paid APIs** (API-Football, Sportmonks, etc.) put player-level shots
  across 8 competitions behind plans of roughly $20–100+/month.
- **FBref**, the original source for this project, stopped being usable
  in 2026. Stats Perform (Opta) ended its data licence in January 2026 and
  had the Opta-sourced stats removed, per-player shots included. The site
  also now serves a Cloudflare challenge that a normal HTTP client can't
  pass (tested from a home connection, September 2026).
- **FotMob** serves the data behind its own site as JSON. One match
  payload has everything needed: minutes, shots, shots on target and xG
  per player; who started; injured and suspended players; the coach;
  formation; team possession. Players and teams have stable numeric IDs.

The trade-off is that these endpoints are **unofficial and undocumented**.
They can change or start refusing requests without notice. Two things
limit the damage. All parsing lives in one function
(`scraper.parse_match_details`). Every raw response is cached, so a
parser fix is applied with `--reparse`, with no re-fetching.

### 2. How scraping stays polite, not evasive

No IP rotation, proxy pools, or anything else meant to hide the scraper
or get around a block. Instead:

- **Two endpoints, few requests.** One request returns a competition's
  whole season fixture list, and one request returns a whole match. A
  weekly update is ~8 fixture-list requests plus one per newly finished
  match.
- **Every finished match is fetched once, ever.** Raw JSON is gzipped to
  `data/raw_cache/` (~27 KB per match) and never requested again.
- **A randomized 2–4 s delay** between requests, and **exponential
  backoff** on 429/5xx.
- **A 403 stops the run.** It means we're being refused, so the job stops
  cleanly instead of retrying. Everything fetched so far is saved, and the
  next run continues from there.
- **A failed or empty match is retried, not forgotten.** A match only
  counts as done once its player stats are stored. Failures are retried on
  later runs, up to `MAX_DETAIL_ATTEMPTS`.

### 3. The weekly cycle

```
┌────────────────────────────────────────────────────────────────┐
│ ONE-TIME, on your PC:  python scripts/backfill.py               │
│   → fixture lists for 2023-24 … 2026-27, all 8 competitions     │
│   → player + team stats for every completed match (~9,000)      │
│   → resumable: Ctrl+C any time, re-run to continue              │
│   → data/matches.db stays local until stage 3 (see Setup)        │
└────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│ GitHub Actions — Mondays 04:00 UTC (also on demand)             │
│                                                                    │
│  1. scripts/run_weekly_scrape.py                                 │
│     → refresh this season's fixture lists (8 requests)           │
│     → fetch stats for newly completed matches (+ earlier fails)  │
│  2. scripts/run_retrain.py                                       │
│     → rebuild features from ALL stored matches, retrain,         │
│       time-based holdout, drift check                            │
│  3. scripts/run_predict.py                                       │
│     → next 7 days' fixtures → likely starters → features →       │
│       scored predictions stored to the database                  │
│  4. commit + push (rewired in stage 3 — see Setup)               │
└────────────────────────────────────────────────────────────────┘
                             │
                             ▼ (push → Streamlit Cloud auto-redeploys)
┌────────────────────────────────────────────────────────────────┐
│ Streamlit app — reads whatever is in the repo right now          │
└────────────────────────────────────────────────────────────────┘
```

### 4. Feature engineering (current — being reworked in stage 2)

For every player, as of the match being predicted. Every feature uses
matches **strictly before** that match's date. Players and opponents are
matched by ID.

| Feature | What it is |
|---|---|
| `overall_avg_shots` / `overall_avg_sot` | Mean over the player's last N games (default 5), following the player across clubs and competitions |
| `overall_sample_size` | How many of those N games actually happened |
| `vs_opp_avg_shots` / `vs_opp_avg_sot` | Mean against this exact opponent (default last 3 meetings), now across seasons |
| `vs_opp_sample_size` | Games behind the vs-opponent number |
| `opp_def_shots_conceded_avg` / `opp_def_sot_conceded_avg` | The opponent's rolling shots/SOT conceded, counting every player who played, substitutes included |
| `opp_def_rating` | The above, z-scored within its competition |
| `is_home` | Home/away flag |

Appearances shorter than `MIN_MINUTES_TO_COUNT` (45) are left out when
features are built. They are still **stored**, so the cut-off can be
changed without re-scraping.

### 5. Why Poisson on top of XGBoost

Shots and SOT are small non-negative counts, which is what a Poisson
distribution models. XGBoost (`count:poisson`) predicts each player's
expected value (λ). The Poisson survival function then gives
`P(shots ≥ k)` for any k, so every threshold comes from one consistent
model. `MIN_PERCENTAGE` is the probability a prediction has to clear to
be flagged "eligible".

### 6. Time-based validation (no leakage)

The most recent `TEST_HOLDOUT_WEEKS` (default 3) are held out of training
and used only for measuring accuracy. The data is never shuffled.

### 7. Drift detection

PSI per feature, compared within the current season. Known limitation:
the season's early weeks have mostly zero-filled form features, so PSI
flags "drift" as sample sizes grow even when nothing real has changed.
Stage 3 replaces this with a calibration check.

### 8. "Likely starters" for upcoming fixtures

Any player who appeared in ≥50% of their team's recent matches. This is a
heuristic, and stated as one. Stage 3 adds re-scoring once FotMob
publishes the confirmed lineups, about an hour before kickoff.

---

## Roadmap

**Stage 1 — data (done).** FotMob client, ID-keyed schema, multi-season
resumable backfill, raw-JSON cache with offline re-parse, failed matches
retried instead of dropped, substitute appearances stored.

**Stage 2 — features that reflect how players actually behave.**
- *Recent form as a core signal:* shots/90 and SOT/90 over the last 4–7
  games, weighted toward the most recent, with recent minutes tracked
  separately so "subbed at 60'" isn't mistaken for "out of form". A
  long-run rate is used only as a fallback when form has 1–2 games behind
  it.
- *Head-to-head across seasons:* team-level H2H (shots for/against in
  the last N meetings, weighted by how recent they were), player-level
  H2H with sample sizes, a derby flag, and opponent style (possession,
  shots conceded) for teams a player has never faced.
- *Circumstances:* rest days, midweek European games, minutes load,
  return from injury, recent manager change, team form, penalty-taker
  status.
- *Evaluation:* a simple baseline (rate × expected minutes × opponent),
  log-loss/Brier and calibration instead of MAE, and ablation tests
  (form windows, with/without H2H) so the data decides what each factor
  is worth.

**Stage 3 — model and app.** Compare GLM / XGBoost / negative binomial,
calibrate, replace PSI with a calibration monitor, re-score with confirmed
lineups, make the deployed app display-only, and decide where the
database lives (see *Failure modes*).

---

## Setup

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. One-time backfill (on your own PC)

```bash
python scripts/backfill.py --dry-run
python scripts/backfill.py
```

`--dry-run` syncs the fixture lists (about 32 requests) and reports how
many matches are left and roughly how long they'll take. A full backfill
of 4 seasons is about 9,000 matches, around **8–9 hours**, so it's best
run overnight. It is safe to stop with Ctrl+C and re-run at any point.
Useful options:

- `--seasons 2025/2026 2026/2027`: only some seasons
- `--competitions "Premier League" "La Liga"`: only some competitions
- `--limit 500`: at most 500 matches this run
- `--reparse`: rebuild the tables from the local cache with no network,
  e.g. after a parser change
- `--retry-given-up`: retry matches that failed 3 times

Why locally and not in Colab: the backfill is slow because of the
deliberate delays, not because it needs computing power. A PC keeps the
cache between runs, doesn't disconnect when idle, and a home connection
is less likely to be refused than a datacenter IP.

### 3. GitHub Actions (the weekly automation)

**Until stage 3, don't commit the full database.** A 4-season backfill
is about 70 MB. That is over GitHub's 50 MB warning, and committing it
every week would add about that much to the repo's history each time.
For now, run the weekly steps on your PC after each round of matches:

```bash
python scripts/run_weekly_scrape.py
python scripts/run_retrain.py
python scripts/run_predict.py
```

Stage 3 rewires `.github/workflows/weekly_retrain.yml`. The database will
live outside git history, and only the small outputs the app displays
(predictions, metrics) will be committed. At that point, check the first
run's log to confirm FotMob answers requests from GitHub's servers. If it
doesn't, a Windows Task Scheduler job on your PC does the same thing.

### 4. Streamlit app

```bash
streamlit run app.py
```

For Streamlit Community Cloud, point it at `app.py`. To use the
"Trigger remote retrain workflow" button, add `GITHUB_TOKEN` (repo scope)
and `GITHUB_REPO` to the app's secrets. The in-app "Retrain now" and
"Fetch" buttons don't persist on Streamlit Cloud, because its filesystem
is a fresh checkout of the repo.

---

## Function reference

### `src/scraper.py`
| Function | Does |
|---|---|
| `fetch_season_fixtures(competition, season)` | One competition-season's full fixture list (played + upcoming), one request |
| `fetch_match_details(match_id)` | Raw match JSON, from cache if we have it; caches finished matches that have stats |
| `parse_match_details(details, match)` | Raw JSON → player rows, team rows, unavailable players |
| `load_cached_details(match_id)` | Read one cached raw payload (no network) |
| `_get_json(...)` | The politeness layer: delay, retry/backoff, 403 → `BlockedError` |

### `src/db.py`
| Function | Does |
|---|---|
| `init_db()` | Creates tables; refuses to run against the old name-keyed schema |
| `upsert_matches(df)` | Fixture sync that never clobbers a match's detail-scrape state |
| `matches_needing_details(...)` | Completed matches still missing stats (incl. retryable failures) |
| `save_match_details(match_id, parsed)` | Stores one match atomically and marks it done |
| `load_player_match_stats` / `load_team_match_stats` / `load_player_unavailability` | Read helpers |
| `scrape_progress()` | Coverage by season / competition / status |
| `save_predictions` / `latest_predictions`, `save_model_metrics`, `save_drift_metrics` | Model outputs |

### `src/pipeline.py`
| Function | Does |
|---|---|
| `sync_fixtures(seasons, competitions)` | Refresh fixture lists; skips finished past seasons already stored |
| `scrape_pending_details(...)` | Resumable detail fetch with progress + ETA; stops cleanly on 403 |
| `reparse_from_cache(seasons)` | Rebuild stats tables from the raw cache, no network |
| `weekly_update()` | The scheduled data step (current season only) |
| `upcoming_fixtures(days_ahead)` | Scheduled matches in the next N days |
| `weekly_retrain(...)` | Full retrain + drift check |
| `generate_weekly_predictions(...)` | Fixtures → likely starters → features → scored predictions |

### `app.py` (Streamlit UI)
| Tab | Does |
|---|---|
| Predictions | This week's player predictions; sidebar sliders re-filter instantly |
| Fetch data | On-demand current-season update + upcoming fixtures |
| Data health | Row counts, holdout MAE trend, drift |
| Retrain | In-session retrain, or trigger the GitHub Actions retrain |

---

## Failure modes worth knowing about

- **FotMob changes its JSON.** Parsing is in one function, and the raw
  cache means a fix only needs `--reparse`. If a key is renamed, matches
  come back `no_stats` or `failed`, and `backfill.py` shows it in its
  progress table.
- **FotMob starts refusing requests (403).** The run stops cleanly with
  exit code 2 and nothing is lost. Try again later, and don't try to get
  around the block.
- **Committing the database weekly.** A 4-season database is about 70 MB,
  measured at ~7 KB per match. Each weekly commit would store a new full
  copy in git history, and GitHub rejects any single file over 100 MB.
  Stage 3 moves the database out of git and commits only the app's small
  outputs.
- **Thin samples.** New signings and early-season weeks have little
  history. Sample-size columns and the low-sample flag in the UI show
  this, rather than presenting those predictions with false confidence.
- **International breaks.** No fixtures in scope means no predictions.
  Use `python scripts/run_predict.py --days 14` to look further ahead.
