"""Entraîne et évalue le modèle avec un découpage chronologique.

Usage :
    python -m prediction.train                      # cible = gagnant
    python -m prediction.train --target place       # cible = placé (top 3)
    python -m prediction.train --target top5        # cible = dans les 5 premiers
    python -m prediction.train --test-start 2026-05-01 --no-market

Le jeu de test correspond aux courses à partir de --test-start : le modèle ne
les voit jamais pendant l'entraînement, comme en conditions réelles.
"""

import os
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from prediction.features import DB_PATH, FEATURES, build_features, load_raw

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))


# Cible -> nombre de chevaux concernés par course (1 gagnant, 3 placés, 5 premiers)
TARGETS = {"win": ("is_win", 1), "place": ("is_place", 3), "top5": ("is_top5", 5)}
N_PLACES = {t: k for t, k in TARGETS.values()}


def race_normalize(df, col):
    return df[col] / df.groupby("course_id")[col].transform("sum")


def evaluate(df, prob_col, target):
    """Métriques par course. df : partants du jeu de test avec une colonne de proba."""
    d = df.copy()
    k = N_PLACES[target]
    # Les probabilités d'une course somment au nombre de chevaux concernés (k, ou moins si peu de partants)
    n = d.groupby("course_id")[prob_col].transform("size").clip(upper=k)
    d["p"] = (race_normalize(d, prob_col) * n).clip(lower=1e-6, upper=1 - 1e-6)
    y = d[target]
    out = {"logloss": -np.mean(y * np.log(d["p"]) + (1 - y) * np.log(1 - d["p"]))}
    # Le cheval jugé le plus probable par course
    top = d.loc[d.groupby("course_id")["p"].idxmax()]
    out["top1_reussite"] = top[target].mean()
    if k == 5:
        # Les 5 chevaux les plus probables du modèle face aux 5 premiers réels
        rk = d.groupby("course_id")["p"].rank(ascending=False, method="first")
        top5 = d[rk <= 5].groupby("course_id")[target]
        out["top5_trouves"] = top5.sum().mean()            # sur 5 en moyenne
        out["top5_complet"] = (top5.sum() >= 5).mean()     # les 5 dans le désordre
    if target == "is_win":
        # ROI pari simple gagnant de 1 € sur le favori du modèle (gain ≈ cote_direct)
        gain = np.where(top["is_win"] == 1, top["cote_direct"].fillna(0), 0)
        out["roi_top1"] = gain.sum() / len(top) - 1
        # Paris « value » : proba du modèle × cote > 1.2
        value = d[(d["p"] * d["cote_direct"] > 1.2) & d["cote_direct"].notna()]
        if len(value):
            g = np.where(value["is_win"] == 1, value["cote_direct"], 0)
            out["roi_value"] = g.sum() / len(value) - 1
            out["nb_paris_value"] = len(value)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--target", choices=list(TARGETS), default="win")
    ap.add_argument("--test-start", default=None, help="AAAA-MM-JJ (défaut : 6 derniers mois)")
    ap.add_argument("--no-market", action="store_true", help="exclure les variables de cote")
    args = ap.parse_args()
    target = TARGETS[args.target][0]

    print("Chargement et calcul des variables…")
    df = build_features(load_raw(args.db))
    df = df[df["ordre_arrivee"].notna() | df["incident"].notna()]  # courses avec résultat
    feats = [f for f in FEATURES if not (args.no_market and (f.startswith("p_ref") or f.startswith("prev_p_ref")))]

    test_start = pd.Timestamp(args.test_start) if args.test_start else df["jour"].max() - pd.DateOffset(months=6)
    # On écarte les 3 premiers mois : historique cheval/driver encore vide
    warmup = df["jour"].min() + pd.DateOffset(months=3)
    train = df[(df["jour"] >= warmup) & (df["jour"] < test_start)]
    test = df[df["jour"] >= test_start].copy()
    feats = [f for f in feats if train[f].nunique() > 1]  # retire les colonnes constantes
    print(f"Entraînement : {train['course_id'].nunique()} courses ({train['jour'].min().date()} → {train['jour'].max().date()})")
    print(f"Test         : {test['course_id'].nunique()} courses ({test['jour'].min().date()} → {test['jour'].max().date()})")

    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, categorical_features="from_dtype",
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=0,
    )
    model.fit(train[feats], train[target])
    print(f"Itérations retenues : {model.n_iter_}")

    test["p_model"] = model.predict_proba(test[feats])[:, 1]
    test["p_uniforme"] = 1.0
    # Comparaison à armes égales : courses où tous les partants ont une cote
    full_ref = test.groupby("course_id")["p_ref"].transform(lambda s: s.notna().all())
    test = test[full_ref].copy()
    print(f"Courses évaluées (toutes cotes connues) : {test['course_id'].nunique()}")
    rows = {"Modèle": evaluate(test, "p_model", target)}
    rows["Marché (cote réf.)"] = evaluate(test, "p_ref", target)
    rows["Hasard"] = evaluate(test, "p_uniforme", target)
    print("\n=== Résultats sur le jeu de test ===")
    print(pd.DataFrame(rows).T.round(4).to_string())

    print("\n=== Par spécialité (réussite du top 1 du modèle) ===")
    for spec, sub in test.groupby("specialite", observed=True):
        m = evaluate(sub, "p_model", target)
        print(f"  {spec:12s} {sub['course_id'].nunique():6d} courses  top1={m['top1_reussite']:.3f}"
              + (f"  roi_top1={m['roi_top1']:+.3f}" if "roi_top1" in m else ""))

    if "quinte" in test and test["quinte"].eq(1).any():
        q = test[test["quinte"] == 1]
        print(f"\n=== Courses Quinté+ ({q['course_id'].nunique()}) ===")
        print(pd.DataFrame({"Modèle": evaluate(q, "p_model", target),
                            "Marché (cote réf.)": evaluate(q, "p_ref", target)}).T.round(4).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    name = f"model_{args.target}{'_nomarket' if args.no_market else ''}.joblib"
    joblib.dump({"model": model, "features": feats, "target": target}, MODELS_DIR / name)
    print(f"\nModèle sauvegardé : models/{name}")


if __name__ == "__main__":
    main()
