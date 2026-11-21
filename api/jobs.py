"""Tâches de fond de l'API.

Usage :
    python -m api.jobs refresh [--date AAAA-MM-JJ]   # calcule et publie les pronostics d'un jour
    python -m api.jobs daily                         # collecte la veille, son bilan, puis pronostics du jour
    python -m api.jobs resultats [--date AAAA-MM-JJ] # copie les arrivées d'un jour collecté (bilan)
    python -m api.jobs bilan [--debut ..] [--fin ..] # affiche le bilan des pronostics archivés
    python -m api.jobs scheduler                     # boucle : daily à 6 h, refresh toutes les 15 min
    python -m api.jobs create-key NOM                # crée une clé d'API pour un utilisateur
    python -m api.jobs list-keys
    python -m api.jobs revoke-key NOM
"""

import argparse
import sqlite3
import time
import traceback

import pandas as pd
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from api import store
from prediction.features import DB_PATH
from prediction.predict import predict_day
from scraper.collect import SCHEMA as PMU_SCHEMA, collect_day
from scraper.pmu_client import PMUClient

PARIS = ZoneInfo("Europe/Paris")
REFRESH_EVERY_MIN = 15
ACTIVE_HOURS = range(7, 23)  # pronostics recalculés de 7 h à 22 h 59 (heure de Paris)
DAILY_HOUR = 6


def today_paris() -> date:
    return datetime.now(PARIS).date()


def refresh(day: date, conn=None, **predict_kwargs) -> int:
    """Calcule les pronostics d'un jour et les publie. Renvoie le nombre de courses."""
    conn = conn or store.connect()
    try:
        df = predict_day(day, **predict_kwargs)
        store.save_predictions(conn, day.isoformat(), df)
        n = df["course_id"].nunique() if not df.empty else 0
        store.log_refresh(conn, day.isoformat(), n, "ok")
        print(f"[{datetime.now(PARIS):%H:%M}] pronostics {day} : {n} courses", flush=True)
        return n
    except Exception as err:
        store.log_refresh(conn, day.isoformat(), 0, "erreur", repr(err))
        raise


def collect_yesterday():
    day = today_paris() - timedelta(days=1)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.executescript(PMU_SCHEMA)
    n = collect_day(PMUClient(), conn, day)
    conn.close()
    print(f"[{datetime.now(PARIS):%H:%M}] collecte {day} : {n} courses", flush=True)
    return day


def import_results(day: date, conn=None, db_path=DB_PATH) -> int:
    """Copie les arrivées d'un jour depuis la base de collecte vers la base de l'API."""
    conn = conn or store.connect()
    pmu = sqlite3.connect(db_path, timeout=60)
    df = pd.read_sql("SELECT course_id, num_pmu, statut, ordre_arrivee, cote_direct FROM participants "
                     "WHERE course_id LIKE ?", pmu, params=(f"{day.isoformat()}_%",))
    pmu.close()
    store.save_results(conn, day.isoformat(), df)
    resume = store.resume_bilan(store.bilan_courses(conn, day.isoformat(), day.isoformat()))
    print(f"[{datetime.now(PARIS):%H:%M}] bilan {day} : {resume}", flush=True)
    return df["course_id"].nunique()


def daily():
    import_results(collect_yesterday())
    refresh(today_paris())


def scheduler():
    """Boucle simple, sans dépendance externe. Une erreur n'arrête pas la boucle."""
    last_daily = None
    last_refresh = 0.0
    print("Planificateur démarré", flush=True)
    while True:
        now = datetime.now(PARIS)
        try:
            if now.hour >= DAILY_HOUR and last_daily != now.date():
                daily()
                last_daily = now.date()
                last_refresh = time.time()
            elif now.hour in ACTIVE_HOURS and time.time() - last_refresh >= REFRESH_EVERY_MIN * 60:
                refresh(now.date())
                last_refresh = time.time()
        except Exception:
            traceback.print_exc()
            last_refresh = time.time()  # on réessaiera au prochain créneau
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh")
    r.add_argument("--date", type=date.fromisoformat, default=None)
    sub.add_parser("daily")
    res = sub.add_parser("resultats")
    res.add_argument("--date", type=date.fromisoformat, default=None)
    b = sub.add_parser("bilan")
    b.add_argument("--debut", type=date.fromisoformat, default=None)
    b.add_argument("--fin", type=date.fromisoformat, default=None)
    sub.add_parser("scheduler")
    k = sub.add_parser("create-key")
    k.add_argument("owner")
    sub.add_parser("list-keys")
    v = sub.add_parser("revoke-key")
    v.add_argument("owner")
    args = ap.parse_args()

    if args.cmd == "refresh":
        refresh(args.date or today_paris())
    elif args.cmd == "daily":
        daily()
    elif args.cmd == "resultats":
        import_results(args.date or today_paris() - timedelta(days=1))
    elif args.cmd == "bilan":
        fin = args.fin or today_paris() - timedelta(days=1)
        debut = args.debut or fin - timedelta(days=29)
        conn = store.connect()
        df = store.bilan_courses(conn, debut.isoformat(), fin.isoformat())
        print(f"Bilan du {debut} au {fin} :", store.resume_bilan(df))
        for jour, g in df.groupby("jour"):
            print(" ", jour, store.resume_bilan(g))
    elif args.cmd == "scheduler":
        scheduler()
    elif args.cmd == "create-key":
        key = store.create_key(store.connect(), args.owner)
        print(f"Clé créée pour {args.owner} (à transmettre, elle ne sera plus affichée) :\n{key}")
    elif args.cmd == "list-keys":
        for row in store.list_keys(store.connect()):
            print(row)
    elif args.cmd == "revoke-key":
        print(f"{store.revoke_owner(store.connect(), args.owner)} clé(s) révoquée(s)")


if __name__ == "__main__":
    main()
