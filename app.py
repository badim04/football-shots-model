"""
Streamlit app — four tabs:
  1. Predictions   — this week's players, adjustable thresholds/confidence
  2. Fetch data    — manual on-demand scrape (the automated weekly job
                      normally does this via GitHub Actions; this is the
                      "run it right now" escape hatch)
  3. Data health   — row counts over time, holdout accuracy trend, drift
  4. Retrain       — manual retrain trigger + retrain history

Run locally with:  streamlit run app.py
"""

import json
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src import db, pipeline

st.set_page_config(page_title="Football Shots Model", layout="wide")
db.init_db()

st.title("⚽ Player Shots & SOT Predictions")
st.caption(
    f"Competitions: {', '.join(config.COMPETITIONS.keys())}  •  "
    f"Predicting {config.CURRENT_SEASON} • history from {', '.join(config.BACKFILL_SEASONS)}"
)

tab_predict, tab_fetch, tab_health, tab_retrain = st.tabs(
    ["📋 Predictions", "🔄 Fetch data", "🩺 Data health", "🧠 Retrain"]
)

# ---------------------------------------------------------------------------
# Shared sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Parameters")
recent_n_overall = st.sidebar.slider("Rolling window — overall form (games)", 2, 10, config.RECENT_N_OVERALL)
recent_n_opponent = st.sidebar.slider("Rolling window — vs. this opponent (games)", 1, 6, config.RECENT_N_OPPONENT)
min_percentage = st.sidebar.slider(
    "Confidence required to flag \"eligible\" (%)", 40, 95, int(config.DEFAULT_MIN_PERCENTAGE * 100)
) / 100
shot_thresholds = st.sidebar.multiselect(
    "Shot thresholds to show", [1, 2, 3, 4], default=config.DEFAULT_SHOT_THRESHOLDS
)
sot_thresholds = st.sidebar.multiselect(
    "Shots-on-target thresholds to show", [1, 2, 3], default=config.DEFAULT_SOT_THRESHOLDS
)
st.sidebar.caption(
    "These control how predictions are FILTERED and DISPLAYED. Changing "
    "the rolling-window sliders only takes effect on the next retrain — "
    "use the Retrain tab after changing them."
)

# ---------------------------------------------------------------------------
# Tab 1: Predictions
# ---------------------------------------------------------------------------
with tab_predict:
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Generate this week's predictions", type="primary"):
            with st.spinner("Scoring upcoming fixtures..."):
                try:
                    result = pipeline.generate_weekly_predictions(
                        min_percentage=min_percentage,
                        shot_thresholds=shot_thresholds or config.DEFAULT_SHOT_THRESHOLDS,
                        sot_thresholds=sot_thresholds or config.DEFAULT_SOT_THRESHOLDS,
                        recent_n_overall=recent_n_overall,
                        recent_n_opponent=recent_n_opponent,
                    )
                    st.session_state["last_predictions"] = result
                except FileNotFoundError:
                    st.error("No trained model found yet — go to the Retrain tab and run a retrain first.")
                except Exception as e:
                    st.error(f"Couldn't generate predictions: {e}")

    preds = st.session_state.get("last_predictions")
    if preds is None:
        stored = db.latest_predictions()
        preds = stored if not stored.empty else None

    if preds is None or preds.empty:
        st.info("No predictions yet. Click **Generate this week's predictions** to run the model "
                "against upcoming fixtures.")
    else:
        # Recompute eligibility live from prob_json so the sidebar sliders
        # work instantly without needing a fresh model run.
        def eligible_any(prob_json_str):
            probs = json.loads(prob_json_str)
            hits = [k for k, v in probs.items() if v >= min_percentage]
            return ", ".join(hits) if hits else "—"

        preds = preds.copy()
        preds["qualifies_for"] = preds["prob_json"].apply(eligible_any)
        preds["has_low_sample"] = (
            (preds["sample_size_overall"] < config.MIN_SAMPLE_SIZE_WARNING)
        )

        only_qualifying = st.checkbox("Show only players who qualify for at least one threshold", value=True)
        show = preds[preds["qualifies_for"] != "—"] if only_qualifying else preds
        show = show.sort_values("pred_shots", ascending=False)

        st.write(f"**{len(show)}** player rows" + (" (filtered)" if only_qualifying else ""))
        display_cols = ["player_name", "team", "opponent", "competition", "date",
                         "pred_shots", "pred_sot", "qualifies_for",
                         "sample_size_overall", "sample_size_vs_opp", "has_low_sample"]
        st.dataframe(
            show[display_cols].rename(columns={
                "sample_size_overall": "games (overall)", "sample_size_vs_opp": "games (vs opp)",
                "has_low_sample": "⚠ low sample",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(
            "⚠ low sample = fewer than "
            f"{config.MIN_SAMPLE_SIZE_WARNING} games of history behind the overall-form number — "
            "treat these predictions as provisional, not because the model failed but because "
            "there isn't enough of this season played yet."
        )

# ---------------------------------------------------------------------------
# Tab 2: Fetch data (manual scrape)
# ---------------------------------------------------------------------------
with tab_fetch:
    st.subheader("Manual data fetch")
    st.markdown(
        "The weekly scrape normally runs automatically (see the GitHub Actions workflow). "
        "Use this if you want fresh data right now — for example, right after this week's "
        "matches finished and before the scheduled job runs. It fetches this season's "
        "newly completed matches from FotMob (plus any that failed on an earlier run).\n\n"
        "**This takes a few minutes** — the scraper waits a few seconds between requests "
        "on purpose. The multi-season history is loaded once with `scripts/backfill.py`, "
        "not from here."
    )
    if st.button("Fetch completed matches now"):
        progress = st.empty()
        progress.info("Fetching... this respects delays between requests, so it will take a few minutes.")
        try:
            summary = pipeline.weekly_update(log=lambda *_: None)["details"]
            if summary["blocked"]:
                progress.warning("FotMob refused requests (403) — stopped early. What was fetched "
                                 "is saved; try again later.")
            else:
                progress.success(
                    f"Done. {summary['matches_to_fetch']} completed matches needed stats: "
                    f"{summary['ok']} stored, {summary['no_stats']} without player stats yet."
                )
            if summary["failures"]:
                st.warning(f"{len(summary['failures'])} match(es) failed — they'll be retried next run.")
        except Exception as e:
            progress.error(f"Fetch failed: {e}")

    st.divider()
    st.subheader("Upcoming fixtures (next 7 days)")
    if st.button("Refresh upcoming fixtures"):
        with st.spinner("Checking upcoming fixtures..."):
            upcoming = pipeline.upcoming_fixtures(7)
            st.session_state["upcoming"] = upcoming
    upcoming = st.session_state.get("upcoming")
    if upcoming is not None and not upcoming.empty:
        st.dataframe(upcoming[["date", "competition", "home_team", "away_team"]],
                     width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab 3: Data health
# ---------------------------------------------------------------------------
with tab_health:
    st.subheader("Is the training data still good?")

    raw = db.load_player_match_stats()
    if raw.empty:
        st.info("No data yet — fetch some matches first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total player-match rows", len(raw))
        c2.metric("Matches covered", raw["match_id"].nunique())
        c3.metric("Date range", f"{raw['date'].min()} → {raw['date'].max()}")

        weekly_counts = (
            raw.assign(week=pd.to_datetime(raw["date"]).dt.to_period("W").astype(str))
            .groupby("week").size().reset_index(name="rows")
        )
        st.plotly_chart(px.bar(weekly_counts, x="week", y="rows", title="Player-match rows scraped per week"),
                         width="stretch")

        st.markdown("#### Holdout accuracy over time")
        metrics_hist = db.load_model_metrics()
        if metrics_hist.empty:
            st.info("No retrains logged yet.")
        else:
            st.plotly_chart(
                px.line(metrics_hist, x="retrain_ts", y=["mae_shots", "mae_sot"],
                        title="Mean absolute error on held-out (most recent) weeks — lower is better"),
                width="stretch",
            )
            st.dataframe(metrics_hist.sort_values("retrain_ts", ascending=False),
                         width="stretch", hide_index=True)

        st.markdown("#### Feature drift (vs. season baseline)")
        drift_hist = db.load_drift_metrics()
        if drift_hist.empty:
            st.info("No drift checks logged yet — these run automatically on every retrain.")
        else:
            latest_ts = drift_hist["retrain_ts"].max()
            latest_drift = drift_hist[drift_hist.retrain_ts == latest_ts]

            def flag_color(f):
                return {"ok": "🟢", "warning": "🟡", "alert": "🔴"}.get(f, "⚪")
            latest_drift = latest_drift.copy()
            latest_drift["status"] = latest_drift["flag"].apply(flag_color)
            st.dataframe(latest_drift[["feature_name", "psi_score", "status"]],
                         width="stretch", hide_index=True)
            if (latest_drift["flag"] == "alert").any():
                st.error(
                    "One or more features have drifted significantly from the season baseline. "
                    "Predictions are still being made, but treat them with more caution until "
                    "this settles — it usually means the league(s) are behaving differently than "
                    "they did early season (new tactical trends, a winter transfer window, etc.)."
                )
            elif (latest_drift["flag"] == "warning").any():
                st.warning("Some features show moderate drift — worth watching, not yet a concern.")
            else:
                st.success("No meaningful drift detected — training data looks consistent with the season baseline.")

            st.plotly_chart(
                px.line(drift_hist, x="retrain_ts", y="psi_score", color="feature_name",
                        title="PSI over time by feature"),
                width="stretch",
            )

# ---------------------------------------------------------------------------
# Tab 4: Retrain
# ---------------------------------------------------------------------------
with tab_retrain:
    st.subheader("Retrain the model")
    st.markdown(
        "This rebuilds every feature from the full database, refits both models "
        "(shots, shots-on-target) with a time-based holdout, and re-runs drift detection. "
        "The automated weekly job does this after each round of matches; use this button "
        "to retrain on demand (e.g. after a manual data fetch, or after changing the "
        "rolling-window sliders)."
    )
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Retrain in this session**")
        st.caption(
            "Retrains immediately using this app's own copy of the database. "
            "Running locally, this is the same files GitHub Actions uses — "
            "fine as your normal workflow. Running on Streamlit Community "
            "Cloud, this app's filesystem is NOT the GitHub repo, so this "
            "updates predictions for you right now but does not persist "
            "back to the repo (the next scheduled Action run will still "
            "pick it up from the real data)."
        )
        if st.button("Retrain now (this session)", type="primary"):
            with st.spinner("Retraining — CPU work, not network, so it's quick (seconds to ~1 min)."):
                try:
                    result = pipeline.weekly_retrain(recent_n_overall, recent_n_opponent)
                    if result["status"] == "no_data":
                        st.warning("No data in the database yet — fetch some matches first.")
                    else:
                        m = result["metrics"]
                        if m["mae_shots"] is not None:
                            st.success(
                                f"Retrained on {m['n_train_rows']} rows "
                                f"(holdout: {m['n_test_rows']} rows, {m['holdout_from']} → {m['holdout_to']}). "
                                f"MAE — shots: {m['mae_shots']:.3f}, SOT: {m['mae_sot']:.3f}"
                            )
                        else:
                            st.success(f"Retrained on {m['n_train_rows']} rows. Not enough data yet for a holdout MAE.")
                except Exception as e:
                    st.error(f"Retrain failed: {e}")

    with colB:
        st.markdown("**Trigger the GitHub Actions retrain**")
        st.caption(
            "Fires the same workflow the weekly schedule uses, on the real "
            "repo. This is the one to use when deployed, if you want the "
            "result to persist and be picked up by everyone. Requires a "
            "`GITHUB_TOKEN` and `GITHUB_REPO` (e.g. `you/football-shots-model`) "
            "set in Streamlit secrets — see the README."
        )
        if st.button("Trigger remote retrain workflow"):
            token = st.secrets.get("GITHUB_TOKEN")
            repo = st.secrets.get("GITHUB_REPO")
            if not token or not repo:
                st.error("GITHUB_TOKEN / GITHUB_REPO not configured in Streamlit secrets — see README.")
            else:
                import requests as _requests
                resp = _requests.post(
                    f"https://api.github.com/repos/{repo}/dispatches",
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                    json={"event_type": "force-retrain"},
                )
                if resp.status_code == 204:
                    st.success("Workflow triggered — check the Actions tab on GitHub for progress "
                               "(usually a few minutes).")
                else:
                    st.error(f"GitHub API returned {resp.status_code}: {resp.text}")

    st.divider()
    metrics_hist = db.load_model_metrics()
    if not metrics_hist.empty:
        st.markdown("#### Retrain history")
        st.dataframe(metrics_hist.sort_values("retrain_ts", ascending=False),
                     width="stretch", hide_index=True)
