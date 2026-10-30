"""
Compare models on the most recent weeks, refit the best, save. Also runnable by hand:
    python scripts/run_retrain.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline

if __name__ == "__main__":
    result = pipeline.weekly_retrain()
    if result["status"] == "no_data":
        print("No data in the database - nothing to train on yet.")
        sys.exit(0)
    rep = result["report"]
    print(f"Holdout {rep['holdout_from']} -> {rep['holdout_to']} ({rep['n_test']:,} rows, "
          f"trained on {rep['n_train']:,})")
    for target, info in rep["targets"].items():
        print(f"\n{target}: chosen {info['chosen']}")
        for kind, m in info["models"].items():
            print(f"  {kind:8s} deviance {m['deviance']:.4f}  NB alpha {m['alpha']:.3f}")
