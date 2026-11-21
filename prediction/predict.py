"""Pronostics pour les courses d'un jour (par défaut aujourd'hui).

Usage :
    python -m prediction.predict
    python -m prediction.predict --date 2026-11-20 --models-dir models

Les partants du jour sont ajoutés à la suite de l'historique de la base pour
calculer les mêmes variables qu'à l'entraînement.
"""

import os
import argparse
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prediction.features import DB_PATH, build_features, load_raw
from scraper.collect import course_row, g, participant_row
from scraper.pmu_client import PMUClient

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))

OUTPUT_COLUMNS = [
    "course_id", "jour", "heure_depart", "hippodrome", "libelle_course", "specialite",
    "distance", "nb_partants", "num_pmu", "nom", "driver", "entraineur",
    "proba", "proba_place", "cote_juste", "cote_reference", "rang", "modele",
]


def fetch_day(day, all_countries=False):
    """Partants (statut PARTANT) de toutes les courses d'un jour, via l'API PMU."""
    client = PMUClient()
    prog = client.programme(day)
    rows = []
    for reunion in (prog or {}).get("reunions", []):
        if not all_countries and g(reunion, "pays", "code") != "FRA":
            continue
        for c in reunion.get("courses", []):
            crow = course_row(day, reunion, c)
            for p in client.participants(day, reunion["numOfficiel"], c["numOrdre"]):
                prow = participant_row(crow["course_id"], p)
                if prow["statut"] != "PARTANT":
                    continue
                prow.update({k: crow[k] for k in (
                    "jour", "heure_depart", "hippodrome", "discipline", "specialite", "distance",
                    "corde", "type_piste", "categorie", "montant_prix", "penetrometre", "temperature")})
                prow["libelle_course"] = crow["libelle"]
                rows.append(prow)
    return pd.DataFrame(rows)


def _predict(bundle, X):
    return bundle["model"].predict_proba(X[bundle["features"]])[:, 1]


def predict_day(day: date, models_dir: Path = MODELS_DIR, db_path: Path = DB_PATH,
                today: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calcule les pronostics d'un jour. `today` permet de fournir les partants
    (tests) au lieu de les télécharger. Renvoie une ligne par partant."""
    if today is None:
        today = fetch_day(day)
    if today.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    today = today.copy()
    hist = load_raw(db_path)
    hist = hist[hist["jour"] < day.isoformat()]  # pas de fuite si la date est passée
    for col in ("ordre_arrivee", "incident", "reduction_km"):
        today[col] = None
    # Aligner les types sur l'historique (les None rendent les colonnes « object »)
    for col in hist.select_dtypes("number").columns:
        if col in today:
            today[col] = pd.to_numeric(today[col], errors="coerce")
    df = build_features(pd.concat([hist, today], ignore_index=True))
    day_df = df[df["jour"] == pd.Timestamp(day)].copy()

    win = joblib.load(models_dir / "model_win.joblib")
    day_df["proba"] = _predict(win, day_df)
    day_df["modele"] = "avec cotes"
    # Courses sans cotes de référence (pas encore publiées) -> modèle sans marché
    no_ref = day_df.groupby("course_id")["p_ref"].transform(lambda s: s.isna().any())
    nomarket_path = models_dir / "model_win_nomarket.joblib"
    if no_ref.any() and nomarket_path.exists():
        nm = joblib.load(nomarket_path)
        day_df.loc[no_ref, "proba"] = _predict(nm, day_df.loc[no_ref])
        day_df.loc[no_ref, "modele"] = "sans cotes"
    day_df["proba"] = day_df["proba"] / day_df.groupby("course_id")["proba"].transform("sum")
    day_df["cote_juste"] = 1 / day_df["proba"]

    place_path = models_dir / "model_place.joblib"
    if place_path.exists():
        pl = _predict(joblib.load(place_path), day_df)
        # Somme des probas de place = nombre de places payées (3, ou moins si peu de partants)
        n_places = day_df["nb_partants"].clip(upper=3)
        pl = pl / day_df.assign(pl=pl).groupby("course_id")["pl"].transform("sum") * n_places
        day_df["proba_place"] = np.clip(pl, 0, 0.99)
    else:
        day_df["proba_place"] = np.nan

    day_df["rang"] = day_df.groupby("course_id")["proba"].rank(ascending=False, method="first").astype(int)
    libelles = today.drop_duplicates("course_id").set_index("course_id")["libelle_course"]
    day_df["libelle_course"] = day_df["course_id"].map(libelles)
    day_df["jour"] = day_df["jour"].dt.date.astype(str)
    day_df["specialite"] = day_df["specialite"].astype(str)
    return day_df[OUTPUT_COLUMNS].sort_values(["heure_depart", "course_id", "rang"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, default=date.today())
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--out", type=Path, default=None, help="export CSV optionnel")
    args = ap.parse_args()

    res = predict_day(args.date, args.models_dir, args.db)
    if res.empty:
        raise SystemExit("Aucune course trouvée pour cette date.")
    for cid, race in res.groupby("course_id", sort=False):
        heure = pd.to_datetime(race["heure_depart"].iloc[0], unit="ms", utc=True).tz_convert("Europe/Paris")
        print(f"\n{cid}  {heure:%H:%M}  {race['hippodrome'].iloc[0]}  {race['libelle_course'].iloc[0]}"
              f"  ({race['specialite'].iloc[0]}, {race['distance'].iloc[0]} m) [modèle {race['modele'].iloc[0]}]")
        for _, r in race.head(5).iterrows():
            cote = f"{r['cote_reference']:.1f}" if pd.notna(r["cote_reference"]) else "  -"
            place = f"  placé={r['proba_place']:.0%}" if pd.notna(r["proba_place"]) else ""
            print(f"   n°{int(r['num_pmu']):>2}  {r['nom'][:24]:24s}  proba={r['proba']:.1%}"
                  f"{place}  cote juste={r['cote_juste']:.1f}  cote PMU={cote}")

    if args.out:
        res.to_csv(args.out, index=False)
        print(f"\nExport : {args.out}")


if __name__ == "__main__":
    main()
