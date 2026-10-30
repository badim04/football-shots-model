"""
Streamlit app — three tabs:
  1. Predictions   — upcoming fixtures' expected starters, adjustable lines/confidence
  2. Track record  — how pre-kickoff predictions actually did + the model report
  3. Admin         — fetch / retrain / predict (only when running locally)

Run locally with:  streamlit run app.py

Deployed (Streamlit Cloud) it reads the small files in data/published/,
because the full database stays on the machine that runs the daily job.
Set SHOTS_READONLY=1 to hide the Admin tab anywhere.
"""

import json
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src import db, evaluate, pipeline

st.set_page_config(page_title="Football Shots Model", layout="wide")

HAVE_DB = os.path.exists(config.DB_PATH)
ADMIN = HAVE_DB and os.environ.get("SHOTS_READONLY") != "1"
if HAVE_DB:
    db.init_db()


def _published(name):
    path = os.path.join(config.PUBLISHED_DIR, name)
    if not os.path.exists(path):
        return None
    if name.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_predictions():
    if HAVE_DB:
        df = db.latest_predictions()
        if not df.empty:
            return df
    return _published("predictions_latest.csv")


st.title("⚽ Player Shots & SOT Predictions")
st.caption(f"{', '.join(config.COMPETITIONS)} • predicting {config.CURRENT_SEASON} • "
           f"history from {config.BACKFILL_SEASONS[0]}")

tabs = st.tabs(["📋 Predictions", "📊 Track record"] + (["🛠 Admin"] if ADMIN else []))

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
min_percentage = st.sidebar.slider("Confidence to count as a pick (%)", 40, 95,
                                   int(config.DEFAULT_MIN_PERCENTAGE * 100)) / 100
lines = st.sidebar.multiselect("Lines", ["shots>=1", "shots>=2", "shots>=3", "sot>=1", "sot>=2"],
                               default=["shots>=1", "shots>=2", "shots>=3", "sot>=1", "sot>=2"])
st.sidebar.caption("These only filter what's shown — the probabilities themselves come from the model.")

# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
with tabs[0]:
    preds = load_predictions()
    if preds is None or preds.empty:
        st.info("No predictions yet — run `python scripts/run_daily.py` on the machine with the database.")
    else:
        preds = preds.copy()
        probs = preds["prob_json"].apply(json.loads)
        for line in lines:
            preds[line] = probs.apply(lambda d: d.get(line))
        preds["qualifies_for"] = probs.apply(
            lambda d: ", ".join(k for k in lines if d.get(k, 0) >= min_percentage) or "—")
        preds["kickoff"] = pd.to_datetime(preds["kickoff_utc"], utc=True, format="ISO8601").dt.strftime("%a %d %b %H:%M")

        c1, c2, c3 = st.columns([1, 1, 2])
        comps = c1.multiselect("Competition", sorted(preds.competition.unique()))
        only = c2.checkbox("Only players with a pick", value=True)
        search = c3.text_input("Player / team contains")
        show = preds
        if comps:
            show = show[show.competition.isin(comps)]
        if only:
            show = show[show.qualifies_for != "—"]
        if search:
            mask = show.player_name.str.contains(search, case=False) | show.team.str.contains(search, case=False)
            show = show[mask]

        st.write(f"**{len(show)}** players · generated {preds.generated_at.max()[:16].replace('T', ' ')} UTC")
        cols = ["kickoff", "competition", "player_name", "team", "opponent", "position",
                "pred_shots", "pred_sot"] + lines + ["qualifies_for", "sample_size_overall",
                                                     "sample_size_vs_opp", "lineup_type"]
        st.dataframe(
            show.sort_values(["kickoff_utc", "pred_shots"], ascending=[True, False])[cols],
            column_config={
                "pred_shots": st.column_config.NumberColumn("xShots", format="%.2f"),
                "pred_sot": st.column_config.NumberColumn("xSOT", format="%.2f"),
                **{line: st.column_config.ProgressColumn(line.replace(">=", " ≥"), min_value=0,
                                                         max_value=1, format="percent") for line in lines},
                "sample_size_overall": "recent games",
                "sample_size_vs_opp": "games vs opp",
                "lineup_type": "lineup",
            },
            width="stretch", hide_index=True,
        )
        st.caption(f"recent games < {config.MIN_SAMPLE_SIZE_WARNING} = thin history, treat as provisional. "
                   "lineup: confirmed / predicted (FotMob) / estimated (from recent starts).")

# ---------------------------------------------------------------------------
# Track record + model report
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("How the published predictions did")
    record = evaluate.track_record(min_percentage) if HAVE_DB else _published("track_record.csv")
    if record is None or record.empty:
        st.info("No completed matches with pre-kickoff predictions yet — this fills in as games are played.")
    else:
        record = record.copy()
        st.dataframe(record.assign(hit_rate=record.hit_rate.map("{:.0%}".format),
                                   avg_prob=record.avg_prob.map("{:.0%}".format)),
                     width="stretch", hide_index=True)
        st.caption("A well-calibrated model's hit rate sits close to its average predicted probability.")
        if HAVE_DB:
            line = st.selectbox("Calibration for line", ["shots>=1", "shots>=2", "sot>=1"])
            cal = evaluate.calibration(line)
            if not cal.empty:
                fig = px.line(cal, x="predicted", y="actual", markers=True, hover_data=["n", "bucket"],
                              title=f"Predicted vs actual — {line}")
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
                st.plotly_chart(fig, width="stretch")
            cov = evaluate.coverage()
            if cov:
                st.write("Expected starters who actually played:",
                         {k: f"{v['played_rate']:.0%} of {v['predicted_players']}" for k, v in cov.items()})

    st.divider()
    st.subheader("Latest model")
    report = db.latest_model_report() if HAVE_DB else _published("model_report.json")
    if not report:
        st.info("No model trained yet.")
    else:
        st.write(f"Holdout {report.get('holdout_from')} → {report.get('holdout_to')} "
                 f"({report.get('n_test', 0):,} starter rows, trained on {report.get('n_train', 0):,}).")
        for target, info in report.get("targets", {}).items():
            st.markdown(f"**{target}** — chosen: `{info['chosen']}`")
            st.dataframe(pd.DataFrame(info["models"]).T.round(4), width="stretch")

# ---------------------------------------------------------------------------
# Admin (local only)
# ---------------------------------------------------------------------------
if ADMIN:
    with tabs[2]:
        st.subheader("Data")
        prog = db.scrape_progress()
        if not prog.empty:
            st.dataframe(prog.pivot_table(index="season", columns="details_status", values="n",
                                          aggfunc="sum", fill_value=0), width="stretch")
        st.caption("The multi-season history comes from `python scripts/backfill.py`; the daily job "
                   "is `python scripts/run_daily.py`. These buttons run the same steps on demand.")
        c1, c2, c3 = st.columns(3)
        if c1.button("Fetch new results"):
            with st.spinner("Fetching (a few seconds per match, on purpose)..."):
                d = pipeline.weekly_update(log=lambda *_: None)["details"]
            st.success(f"{d['ok']} new match(es) stored." + (" FotMob refused requests - stopped early."
                                                            if d["blocked"] else ""))
        if c2.button("Retrain"):
            with st.spinner("Comparing models and retraining..."):
                r = pipeline.weekly_retrain()
            if r["status"] == "ok":
                st.success({t: v["chosen"] for t, v in r["report"]["targets"].items()})
            else:
                st.warning(r["status"])
        if c3.button("Predict next 7 days"):
            with st.spinner("Fetching expected lineups and scoring..."):
                p = pipeline.generate_weekly_predictions(log=lambda *_: None)
                pipeline.publish_files(p)
            load_predictions.clear()
            st.success(f"{len(p)} predictions across {p.match_id.nunique() if not p.empty else 0} fixtures.")
