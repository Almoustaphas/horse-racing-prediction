"""Compare des jeux de variables sur plusieurs périodes de test successives
(validation « walk-forward »), pour juger d'une amélioration de façon fiable.

Usage :
    python -m prediction.experiment
    python -m prediction.experiment --folds 3 --months 4 --target win
"""

import argparse
import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from prediction.features import DB_PATH, FEATURES, FEATURES_V1, build_features, load_raw
from prediction.train import evaluate

MARKET = lambda f: f.startswith("p_ref") or f.startswith("prev_p_ref")


def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, categorical_features="from_dtype",
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--months", type=int, default=4)
    ap.add_argument("--target", choices=["win", "place"], default="win")
    args = ap.parse_args()
    target = "is_win" if args.target == "win" else "is_place"

    t0 = time.time()
    df = build_features(load_raw(DB_PATH))
    df = df[df["ordre_arrivee"].notna() | df["incident"].notna()]
    full_ref = df.groupby("course_id")["p_ref"].transform(lambda s: s.notna().all())
    print(f"Variables calculées en {time.time() - t0:.0f} s")

    sets = {
        "v1": FEATURES_V1,
        "v2": FEATURES,
        "v1 sans cotes": [f for f in FEATURES_V1 if not MARKET(f)],
        "v2 sans cotes": [f for f in FEATURES if not MARKET(f)],
    }
    end = df["jour"].max()
    warmup = df["jour"].min() + pd.DateOffset(months=3)
    results = []
    for i in range(args.folds, 0, -1):
        t_start = end - pd.DateOffset(months=args.months * i)
        t_end = end - pd.DateOffset(months=args.months * (i - 1))
        train = df[(df["jour"] >= warmup) & (df["jour"] < t_start)]
        test = df[(df["jour"] >= t_start) & (df["jour"] < t_end + pd.Timedelta(days=1)) & full_ref].copy()
        label = f"{t_start.date()}→{t_end.date()}"
        res = {"Marché": evaluate(test, "p_ref", target)}
        for name, feats in sets.items():
            feats = [f for f in feats if train[f].nunique() > 1]
            m = make_model().fit(train[feats], train[target])
            test["p"] = m.predict_proba(test[feats])[:, 1]
            res[name] = evaluate(test, "p", target)
        for name, r in res.items():
            results.append({"periode": label, "jeu": name, **r})
        print(f"\n{label}  ({test['course_id'].nunique()} courses)")
        print(pd.DataFrame(res).T.round(4).to_string())

    summary = pd.DataFrame(results).groupby("jeu").mean(numeric_only=True)
    print("\n=== Moyenne sur les périodes ===")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
