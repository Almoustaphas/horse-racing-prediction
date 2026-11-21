"""Stockage de l'API : pronostics pré-calculés, archive, résultats, clés d'API, journal.

Base SQLite séparée de la base de collecte (database/pmu.sqlite) : l'API ne fait
que lire des pronostics déjà calculés et n'interroge jamais le PMU.
"""

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
API_DB_PATH = Path(os.environ.get("API_DB_PATH", ROOT / "database" / "api.sqlite"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS pronostics (
    course_id TEXT, num_pmu INTEGER, jour TEXT, heure_depart INTEGER,
    hippodrome TEXT, libelle_course TEXT, specialite TEXT, distance INTEGER,
    nb_partants INTEGER, nom TEXT, driver TEXT, entraineur TEXT,
    proba REAL, proba_place REAL, cote_juste REAL, cote_reference REAL,
    rang INTEGER, modele TEXT, updated_at TEXT,
    PRIMARY KEY (course_id, num_pmu)
);
CREATE INDEX IF NOT EXISTS idx_prono_jour ON pronostics(jour);
-- Dernier pronostic publié AVANT le départ de chaque course : il n'est plus modifié
-- une fois la course partie et sert à mesurer la performance réelle.
CREATE TABLE IF NOT EXISTS pronostics_archive (
    course_id TEXT, num_pmu INTEGER, jour TEXT, heure_depart INTEGER,
    hippodrome TEXT, libelle_course TEXT, specialite TEXT, distance INTEGER,
    nb_partants INTEGER, nom TEXT, driver TEXT, entraineur TEXT,
    proba REAL, proba_place REAL, cote_juste REAL, cote_reference REAL,
    rang INTEGER, modele TEXT, updated_at TEXT,
    PRIMARY KEY (course_id, num_pmu)
);
CREATE INDEX IF NOT EXISTS idx_archive_jour ON pronostics_archive(jour);
-- Arrivées officielles (copiées depuis la base de collecte)
CREATE TABLE IF NOT EXISTS resultats (
    course_id TEXT, num_pmu INTEGER, jour TEXT, statut TEXT,
    ordre_arrivee INTEGER, cote_direct REAL,
    PRIMARY KEY (course_id, num_pmu)
);
CREATE INDEX IF NOT EXISTS idx_resultats_jour ON resultats(jour);
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash TEXT PRIMARY KEY, owner TEXT NOT NULL, created_at TEXT, active INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS refresh_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, jour TEXT, ran_at TEXT,
    nb_courses INTEGER, statut TEXT, message TEXT
);
"""


ARCHIVE_COLS = ("course_id, num_pmu, jour, heure_depart, hippodrome, libelle_course, specialite, "
                "distance, nb_partants, nom, driver, entraineur, proba, proba_place, cote_juste, "
                "cote_reference, rang, modele, updated_at")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or API_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # lectures possibles pendant une écriture
    conn.executescript(SCHEMA)
    with conn:  # reprise des pronostics publiés avant l'existence de l'archive
        conn.execute(f"""
            INSERT OR IGNORE INTO pronostics_archive ({ARCHIVE_COLS})
            SELECT {ARCHIVE_COLS} FROM pronostics
            WHERE heure_depart > CAST(strftime('%s', updated_at) AS INTEGER) * 1000
              AND course_id NOT IN (SELECT DISTINCT course_id FROM pronostics_archive)""")
    return conn


# ------------------------------------------------------------- pronostics
def _rows(df, cols):
    return [tuple(None if pd.isna(v) else v for v in r) for r in df[cols].itertuples(index=False)]


def save_predictions(conn, day: str, df: pd.DataFrame, now_ms: int | None = None):
    """Remplace atomiquement les pronostics d'un jour.

    Les courses pas encore parties sont aussi copiées dans l'archive (en remplaçant
    la version précédente) ; celles déjà parties gardent leur dernière version d'avant départ.
    """
    df = df.copy()
    df["updated_at"] = now_iso()
    now_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    cols = list(df.columns)
    insert = "INSERT INTO {} (%s) VALUES (%s)" % (",".join(cols), ",".join("?" * len(cols)))
    a_venir = df[df["heure_depart"] > now_ms] if not df.empty else df
    ids = a_venir["course_id"].unique().tolist() if not df.empty else []
    with conn:
        conn.execute("DELETE FROM pronostics WHERE jour = ?", (day,))
        conn.executemany(insert.format("pronostics"), _rows(df, cols))
        conn.executemany("DELETE FROM pronostics_archive WHERE course_id = ?", [(i,) for i in ids])
        conn.executemany(insert.format("pronostics_archive"), _rows(a_venir, cols))


def log_refresh(conn, day, nb_courses, statut, message=""):
    with conn:
        conn.execute("INSERT INTO refresh_log (jour, ran_at, nb_courses, statut, message) VALUES (?,?,?,?,?)",
                     (day, now_iso(), nb_courses, statut, message[:500]))


def last_refresh(conn):
    row = conn.execute("SELECT * FROM refresh_log ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def get_day(conn, day: str):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM pronostics WHERE jour = ? ORDER BY heure_depart, course_id, rang", (day,))]


def get_course(conn, course_id: str):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM pronostics WHERE course_id = ? ORDER BY rang", (course_id,))]


# ------------------------------------------------------------- bilan
def save_results(conn, day: str, df: pd.DataFrame):
    """Remplace les arrivées d'un jour (colonnes : course_id, num_pmu, statut, ordre_arrivee, cote_direct)."""
    df = df.assign(jour=day)
    cols = ["course_id", "num_pmu", "jour", "statut", "ordre_arrivee", "cote_direct"]
    with conn:
        conn.execute("DELETE FROM resultats WHERE jour = ?", (day,))
        conn.executemany(f"INSERT INTO resultats ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                         _rows(df, cols))


def bilan_courses(conn, debut: str, fin: str) -> pd.DataFrame:
    """Une ligne par course archivée et arrivée : premier choix du modèle et favori du marché.

    Une course est écartée si elle n'a pas de gagnant connu ou si le premier choix
    du modèle n'a pas pris le départ (mise remboursée).
    """
    a = pd.read_sql("SELECT course_id, jour, num_pmu, rang, proba, cote_reference, modele "
                    "FROM pronostics_archive WHERE jour BETWEEN ? AND ?", conn, params=(debut, fin))
    r = pd.read_sql("SELECT course_id, num_pmu, statut, ordre_arrivee, cote_direct "
                    "FROM resultats WHERE jour BETWEEN ? AND ?", conn, params=(debut, fin))
    cols = ["course_id", "jour", "modele", "top1_num", "top1_gagne", "top1_place", "top1_cote",
            "top1_gain", "favori_gagne"]
    if a.empty or r.empty:
        return pd.DataFrame(columns=cols)
    m = a.merge(r, on=["course_id", "num_pmu"], how="left")
    gagnees = set(r.loc[r["ordre_arrivee"] == 1, "course_id"])
    out = []
    for cid, g in m.groupby("course_id"):
        if cid not in gagnees:
            continue
        top = g.loc[g["rang"].idxmin()]
        if top["statut"] != "PARTANT":
            continue
        cotes = g.loc[g["statut"] == "PARTANT", ["cote_reference", "ordre_arrivee"]].dropna(subset=["cote_reference"])
        cotes = cotes[cotes["cote_reference"] > 1]
        gagne = top["ordre_arrivee"] == 1
        cote = top["cote_direct"] if pd.notna(top["cote_direct"]) and top["cote_direct"] > 1 else None
        out.append({
            "course_id": cid, "jour": top["jour"], "modele": top["modele"], "top1_num": int(top["num_pmu"]),
            "top1_gagne": bool(gagne), "top1_place": bool(1 <= (top["ordre_arrivee"] or 0) <= 3),
            "top1_cote": cote, "top1_gain": (cote if gagne else 0.0) if cote else None,
            "favori_gagne": (bool(cotes.loc[cotes["cote_reference"].idxmin(), "ordre_arrivee"] == 1)
                             if not cotes.empty else None),
        })
    return pd.DataFrame(out, columns=cols)


def resume_bilan(df: pd.DataFrame) -> dict:
    """Agrège un bilan : réussite du premier choix, ROI au simple gagnant (1 € par course)."""
    paris = df.dropna(subset=["top1_gain"])
    fav = df.dropna(subset=["favori_gagne"])
    mises = len(paris)
    gains = float(paris["top1_gain"].sum()) if mises else 0.0
    return {
        "courses": int(len(df)),
        "reussite_premier_choix": float(df["top1_gagne"].mean()) if len(df) else None,
        "premier_choix_dans_les_3": float(df["top1_place"].mean()) if len(df) else None,
        "reussite_favori_marche": float(fav["favori_gagne"].astype(bool).mean()) if len(fav) else None,
        "mises": mises,
        "gains": round(gains, 2),
        "roi": round(gains / mises - 1, 4) if mises else None,
    }


# ------------------------------------------------------------- clés d'API
def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(conn, owner: str) -> str:
    """Crée une clé et renvoie sa valeur en clair (seul son hash est stocké)."""
    key = "hrp_" + secrets.token_urlsafe(24)
    with conn:
        conn.execute("INSERT INTO api_keys (key_hash, owner, created_at) VALUES (?,?,?)",
                     (_hash(key), owner, now_iso()))
    return key


def check_key(conn, key: str | None) -> str | None:
    """Renvoie le propriétaire si la clé est valide et active, sinon None."""
    if not key:
        return None
    row = conn.execute("SELECT owner FROM api_keys WHERE key_hash = ? AND active = 1",
                       (_hash(key),)).fetchone()
    return row["owner"] if row else None


def list_keys(conn):
    return [dict(r) for r in conn.execute(
        "SELECT owner, created_at, active, substr(key_hash, 1, 12) AS empreinte FROM api_keys ORDER BY created_at")]


def revoke_owner(conn, owner: str) -> int:
    with conn:
        return conn.execute("UPDATE api_keys SET active = 0 WHERE owner = ?", (owner,)).rowcount
