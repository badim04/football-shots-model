"""
Settle modelling questions with held-out data instead of opinion:

    python scripts/evaluate.py                 # model comparison + feature-group ablation
    python scripts/evaluate.py --halflives 1 2 3 5 8    # how "recent" should form be?

Every number is on the most recent TEST_HOLDOUT_WEEKS, with models fitted
only on earlier matches. Lower deviance / log-loss = better. "Δ vs all"
is how much WORSE the model gets when that group of features is removed —
a big positive number means the group carries real information.
Writes a markdown report to logs/.
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import db, features, train


def _fmt(results, target):
    lines = []
    for kind, m in results["targets"][target].items():
        lls = "  ".join(f"{k} {v:.4f}" for k, v in m.items() if k.startswith("logloss"))
        lines.append(f"| {kind} | {m['deviance']:.4f} | {lls} | {m['alpha']:.3f} |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="glm", choices=["glm", "xgb"], help="model used for ablations")
    ap.add_argument("--halflives", nargs="*", type=float, help="form half-lives (games) to compare")
    args = ap.parse_args()

    players, teams = db.load_player_match_stats(), db.load_team_match_stats()
    feat = features.build_features(players, teams)
    rows = features.training_rows(feat)
    print(f"{len(rows):,} training rows, {rows.match_id.nunique():,} matches, "
          f"{rows.season.nunique()} season(s)")

    out = [f"# Evaluation — {datetime.now():%Y-%m-%d %H:%M}", "",
           f"{len(rows):,} starter rows from {rows.match_id.nunique():,} matches "
           f"({', '.join(sorted(rows.season.unique()))}).", ""]

    base = train.compare_models(rows)
    base.pop("_test")
    out += [f"Holdout: {base['holdout_from']} → {base['holdout_to']} ({base['n_test']:,} rows).", ""]
    for target in train.TARGETS:
        out += [f"## Models — {target}", "", "| model | deviance | log-loss by line | NB alpha |",
                "|---|---|---|---|"] + _fmt(base, target) + [""]

    ref = {t: base["targets"][t][args.kind]["deviance"] for t in train.TARGETS}
    out += [f"## Feature groups ({args.kind}) — what each one is worth", "",
            "| removed | Δ shots deviance | Δ SOT deviance |", "|---|---|---|"]
    for group, cols in features.FEATURE_GROUPS.items():
        keep = [c for c in features.FEATURE_COLUMNS if c not in cols]
        r = train.compare_models(rows, kinds=(args.kind,), features=keep)
        d = {t: r["targets"][t][args.kind]["deviance"] - ref[t] for t in train.TARGETS}
        out.append(f"| {group} | {d['shots']:+.4f} | {d['sot']:+.4f} |")
        print(f"without {group:14s} shots {d['shots']:+.4f}  sot {d['sot']:+.4f}")
    out.append("")

    if args.halflives:
        out += [f"## Form half-life ({args.kind})", "", "| half-life (games) | shots deviance | SOT deviance |",
                "|---|---|---|"]
        original = config.FORM_HALFLIFE_GAMES
        for hl in args.halflives:
            config.FORM_HALFLIFE_GAMES = hl
            r = train.compare_models(features.training_rows(features.build_features(players, teams)),
                                     kinds=(args.kind,))
            dv = {t: r["targets"][t][args.kind]["deviance"] for t in train.TARGETS}
            out.append(f"| {hl:g} | {dv['shots']:.4f} | {dv['sot']:.4f} |")
            print(f"half-life {hl:g}: shots {dv['shots']:.4f}  sot {dv['sot']:.4f}")
        config.FORM_HALFLIFE_GAMES = original

    path = os.path.join(config.LOGS_DIR, f"evaluation_{datetime.now():%Y%m%d_%H%M}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
