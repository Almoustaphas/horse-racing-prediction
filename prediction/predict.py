"""Pronostics pour les courses d'un jour (par défaut aujourd'hui).

Usage :
    python -m prediction.predict
    python -m prediction.predict --date 2026-11-20 --model models/model_win.joblib

Les partants du jour sont ajoutés à la suite de l'historique de la base pour
calculer les mêmes variables qu'à l'entraînement.
"""

import argparse
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from prediction.features import DB_PATH, build_features, load_raw
from scraper.collect import course_row, g, participant_row
from scraper.pmu_client import PMUClient

ROOT = Path(__file__).resolve().parent.parent


def fetch_day(day, all_countries=False):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, default=date.today())
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "model_win.joblib")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--out", type=Path, default=None, help="export CSV optionnel")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    today = fetch_day(args.date)
    if today.empty:
        raise SystemExit("Aucune course trouvée pour cette date.")
    hist = load_raw(args.db)
    hist = hist[hist["jour"] < args.date.isoformat()]  # pas de fuite si la date est passée
    today["ordre_arrivee"] = None
    today["incident"] = None
    today["reduction_km"] = None
    df = build_features(pd.concat([hist, today], ignore_index=True))
    day_df = df[df["jour"] == pd.Timestamp(args.date)].copy()

    day_df["proba"] = bundle["model"].predict_proba(day_df[bundle["features"]])[:, 1]
    day_df["proba"] = day_df["proba"] / day_df.groupby("course_id")["proba"].transform("sum")
    day_df["cote_juste"] = 1 / day_df["proba"]

    libelles = today.drop_duplicates("course_id").set_index("course_id")["libelle_course"]
    for cid, race in day_df.sort_values("heure_depart").groupby("course_id", sort=False):
        heure = pd.to_datetime(race["heure_depart"].iloc[0], unit="ms", utc=True).tz_convert("Europe/Paris")
        print(f"\n{cid}  {heure:%H:%M}  {race['hippodrome'].iloc[0]}  {libelles.get(cid, '')}"
              f"  ({race['specialite'].iloc[0]}, {race['distance'].iloc[0]} m)")
        top = race.sort_values("proba", ascending=False).head(5)
        for _, r in top.iterrows():
            cote = f"{r['cote_reference']:.1f}" if pd.notna(r["cote_reference"]) else "  -"
            print(f"   n°{int(r['num_pmu']):>2}  {r['nom'][:24]:24s}  proba={r['proba']:.1%}"
                  f"  cote juste={r['cote_juste']:.1f}  cote PMU={cote}")

    if args.out:
        day_df[["course_id", "num_pmu", "nom", "proba", "cote_juste", "cote_reference"]].to_csv(args.out, index=False)
        print(f"\nExport : {args.out}")


if __name__ == "__main__":
    main()
