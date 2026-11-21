"""Collecte l'historique PMU dans une base SQLite.

Usage :
    python -m scraper.collect --start 2023-01-01 --end 2026-11-18
    python -m scraper.collect --start 2026-11-15 --end 2026-11-15 --all-countries
    python -m scraper.collect --backfill-quinte   # Quinté+ des jours déjà collectés

La collecte est reprenable : un jour déjà présent dans `collected_days`
est ignoré (sauf --force).
"""

import os
import argparse
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

from scraper.pmu_client import PMUClient

DB_PATH = Path(os.environ.get("PMU_DB_PATH", Path(__file__).resolve().parent.parent / "database" / "pmu.sqlite"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS collected_days (
    jour TEXT PRIMARY KEY,
    nb_courses INTEGER,
    collected_at TEXT
);
CREATE TABLE IF NOT EXISTS courses (
    course_id TEXT PRIMARY KEY,          -- AAAA-MM-JJ_R1_C1
    jour TEXT, num_reunion INTEGER, num_course INTEGER,
    heure_depart INTEGER,                -- timestamp ms
    hippodrome TEXT, hippodrome_libelle TEXT, pays TEXT,
    libelle TEXT, discipline TEXT, specialite TEXT,
    distance INTEGER, corde TEXT, type_piste TEXT,
    categorie TEXT, condition_age TEXT, condition_sexe TEXT,
    montant_prix INTEGER, nb_declares INTEGER,
    penetrometre REAL, terrain TEXT,
    meteo TEXT, temperature REAL, vent REAL,
    statut TEXT, duree_course INTEGER, nature_reunion TEXT,
    quinte INTEGER                       -- 1 si course support du Quinté+
);
CREATE TABLE IF NOT EXISTS participants (
    course_id TEXT, num_pmu INTEGER,
    id_cheval TEXT, nom TEXT, age INTEGER, sexe TEXT, race TEXT, pays_cheval TEXT,
    statut TEXT, oeilleres TEXT, deferre TEXT,
    driver TEXT, driver_change INTEGER, entraineur TEXT, proprietaire TEXT, eleveur TEXT,
    pere TEXT, mere TEXT, pere_mere TEXT,
    musique TEXT, inedit INTEGER,
    nb_courses INTEGER, nb_victoires INTEGER, nb_places INTEGER,
    nb_places_2 INTEGER, nb_places_3 INTEGER,
    gains_carriere INTEGER, gains_victoires INTEGER, gains_place INTEGER,
    gains_annee INTEGER, gains_annee_prec INTEGER,
    handicap_valeur REAL, handicap_poids INTEGER, handicap_distance INTEGER,
    place_corde INTEGER, jument_pleine INTEGER, supplement INTEGER,
    cote_reference REAL,                 -- cote de référence (avant course)
    cote_direct REAL,                    -- dernière cote (≈ cote au départ)
    favori INTEGER,
    -- résultats (NE PAS utiliser comme variables d'entrée)
    ordre_arrivee INTEGER, incident TEXT,
    temps_obtenu INTEGER, reduction_km INTEGER, ecart TEXT,
    PRIMARY KEY (course_id, num_pmu)
);
CREATE INDEX IF NOT EXISTS idx_part_cheval ON participants(id_cheval);
CREATE INDEX IF NOT EXISTS idx_courses_jour ON courses(jour);
"""


def g(d, *keys):
    """Accès imbriqué tolérant : g(x, 'a', 'b') == x['a']['b'] ou None."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def to_float(v):
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def is_quinte(c) -> int:
    """1 si la course est support du Quinté+ (pari QUINTE_PLUS proposé)."""
    return int(any("QUINTE" in (p.get("typePari") or "") for p in c.get("paris") or []))


def init_db(conn):
    """Crée les tables et ajoute les colonnes apparues depuis (bases existantes)."""
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(courses)")}
    if "quinte" not in cols:
        conn.execute("ALTER TABLE courses ADD COLUMN quinte INTEGER")
        conn.commit()


def backfill_quinte(client, conn, workers=4):
    """Renseigne la colonne quinte des jours collectés avant son ajout (1 requête par jour)."""
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT jour FROM courses WHERE quinte IS NULL ORDER BY jour")]
    print(f"{len(days)} jours à compléter", flush=True)

    def fetch(day):
        return day, client.programme(date.fromisoformat(day))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (day, prog) in enumerate(pool.map(fetch, days), 1):
            rows = [(is_quinte(c), f"{day}_R{r['numOfficiel']}_C{c['numOrdre']}")
                    for r in (prog or {}).get("reunions", []) for c in r.get("courses", [])]
            conn.executemany("UPDATE courses SET quinte = ? WHERE course_id = ?", rows)
            conn.execute("UPDATE courses SET quinte = 0 WHERE jour = ? AND quinte IS NULL", (day,))
            conn.commit()
            if i % 100 == 0:
                print(f"  {i}/{len(days)} ({day})", flush=True)


def course_row(day, reunion, c):
    return {
        "course_id": f"{day.isoformat()}_R{reunion['numOfficiel']}_C{c['numOrdre']}",
        "jour": day.isoformat(),
        "num_reunion": reunion["numOfficiel"],
        "num_course": c["numOrdre"],
        "heure_depart": c.get("heureDepart"),
        "hippodrome": g(reunion, "hippodrome", "code"),
        "hippodrome_libelle": g(reunion, "hippodrome", "libelleCourt"),
        "pays": g(reunion, "pays", "code"),
        "libelle": c.get("libelle"),
        "discipline": c.get("discipline"),
        "specialite": c.get("specialite"),
        "distance": c.get("distance"),
        "corde": c.get("corde"),
        "type_piste": c.get("typePiste"),
        "categorie": c.get("categorieParticularite"),
        "condition_age": c.get("conditionAge"),
        "condition_sexe": c.get("conditionSexe"),
        "montant_prix": c.get("montantPrix"),
        "nb_declares": c.get("nombreDeclaresPartants"),
        "penetrometre": to_float(g(c, "penetrometre", "valeurMesure")),
        "terrain": g(c, "penetrometre", "intitule"),
        "meteo": g(reunion, "meteo", "nebulositeLibelleCourt"),
        "temperature": g(reunion, "meteo", "temperature"),
        "vent": g(reunion, "meteo", "forceVent"),
        "statut": c.get("statut"),
        "duree_course": c.get("dureeCourse"),
        "nature_reunion": reunion.get("nature"),
        "quinte": is_quinte(c),
    }


def participant_row(course_id, p):
    return {
        "course_id": course_id,
        "num_pmu": p.get("numPmu"),
        "id_cheval": p.get("idCheval"),
        "nom": p.get("nom"),
        "age": p.get("age"),
        "sexe": p.get("sexe"),
        "race": p.get("race"),
        "pays_cheval": p.get("pays"),
        "statut": p.get("statut"),
        "oeilleres": p.get("oeilleres"),
        "deferre": p.get("deferre"),
        "driver": p.get("driver"),
        "driver_change": int(bool(p.get("driverChange"))),
        "entraineur": p.get("entraineur"),
        "proprietaire": p.get("proprietaire"),
        "eleveur": p.get("eleveur"),
        "pere": p.get("nomPere"),
        "mere": p.get("nomMere"),
        "pere_mere": p.get("nomPereMere"),
        "musique": p.get("musique"),
        "inedit": int(bool(p.get("indicateurInedit"))),
        "nb_courses": p.get("nombreCourses"),
        "nb_victoires": p.get("nombreVictoires"),
        "nb_places": p.get("nombrePlaces"),
        "nb_places_2": p.get("nombrePlacesSecond"),
        "nb_places_3": p.get("nombrePlacesTroisieme"),
        "gains_carriere": g(p, "gainsParticipant", "gainsCarriere"),
        "gains_victoires": g(p, "gainsParticipant", "gainsVictoires"),
        "gains_place": g(p, "gainsParticipant", "gainsPlace"),
        "gains_annee": g(p, "gainsParticipant", "gainsAnneeEnCours"),
        "gains_annee_prec": g(p, "gainsParticipant", "gainsAnneePrecedente"),
        "handicap_valeur": p.get("handicapValeur"),
        "handicap_poids": p.get("handicapPoids"),
        "handicap_distance": p.get("handicapDistance"),
        "place_corde": p.get("placeCorde"),
        "jument_pleine": int(bool(p.get("jumentPleine"))),
        "supplement": p.get("supplement"),
        "cote_reference": g(p, "dernierRapportReference", "rapport"),
        "cote_direct": g(p, "dernierRapportDirect", "rapport"),
        "favori": int(bool(g(p, "dernierRapportDirect", "favoris"))),
        "ordre_arrivee": p.get("ordreArrivee"),
        "incident": p.get("incident"),
        "temps_obtenu": p.get("tempsObtenu"),
        "reduction_km": p.get("reductionKilometrique"),
        "ecart": g(p, "distanceChevalPrecedent", "libelleCourt"),
    }


def upsert(conn, table, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])


def collect_day(client, conn, day, all_countries=False, workers=4):
    prog = client.programme(day)
    if not prog:
        return 0
    jobs = []
    for reunion in prog.get("reunions", []):
        if not all_countries and g(reunion, "pays", "code") != "FRA":
            continue
        for c in reunion.get("courses", []):
            # On ne garde que les courses courues avec une arrivée
            # (statut ARRIVEE_DEFINITIVE_* ou FIN_COURSE selon les années)
            if c.get("categorieStatut") != "ARRIVEE" or not c.get("ordreArrivee"):
                continue
            jobs.append((reunion, c))

    def fetch(job):
        reunion, c = job
        return job, client.participants(day, reunion["numOfficiel"], c["numOrdre"])

    course_rows, part_rows = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (reunion, c), parts in pool.map(fetch, jobs):
            row = course_row(day, reunion, c)
            course_rows.append(row)
            part_rows.extend(participant_row(row["course_id"], p) for p in parts)

    upsert(conn, "courses", course_rows)
    upsert(conn, "participants", part_rows)
    conn.execute(
        "INSERT OR REPLACE INTO collected_days VALUES (?,?,?)",
        (day.isoformat(), len(course_rows), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    return len(course_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--all-countries", action="store_true", help="inclure les courses étrangères")
    ap.add_argument("--force", action="store_true", help="recollecter les jours déjà présents")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backfill-quinte", action="store_true",
                    help="renseigner le Quinté+ des jours déjà collectés, puis quitter")
    args = ap.parse_args()
    if not args.start and not args.backfill_quinte:
        ap.error("--start est obligatoire")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db, check_same_thread=False, timeout=120)
    init_db(conn)
    if args.backfill_quinte:
        backfill_quinte(PMUClient(), conn, args.workers)
        conn.close()
        return
    done = {r[0] for r in conn.execute("SELECT jour FROM collected_days")}

    client = PMUClient()
    day = args.start
    while day <= args.end:
        if args.force or day.isoformat() not in done:
            try:
                n = collect_day(client, conn, day, args.all_countries, args.workers)
                print(f"{day} : {n} courses", flush=True)
            except Exception as err:  # on continue, le jour sera retenté au prochain lancement
                print(f"{day} : ERREUR {err}", flush=True)
        day += timedelta(days=1)
    conn.close()


if __name__ == "__main__":
    main()
