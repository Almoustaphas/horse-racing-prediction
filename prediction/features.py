"""Construction des variables (features) par partant.

Règle d'or : chaque variable ne doit utiliser QUE de l'information disponible
avant le départ de la course. Les statistiques historiques (cheval, driver,
entraîneur) sont calculées en cumul sur les courses antérieures uniquement
(tri chronologique + exclusion de la ligne courante).
"""

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "database" / "pmu.sqlite"

MUSIQUE_TOKEN = re.compile(r"(\d|[A-Z])([a-z])")

# Variables catégorielles encodées par le modèle
CATEGORICAL = ["specialite", "sexe", "oeilleres", "deferre", "corde", "type_piste", "categorie"]


def load_raw(db_path=DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT p.*, c.jour, c.heure_depart, c.hippodrome, c.discipline, c.specialite,
               c.distance, c.corde, c.type_piste, c.categorie, c.montant_prix,
               c.penetrometre, c.temperature
        FROM participants p JOIN courses c USING (course_id)
        WHERE p.statut = 'PARTANT'
        """,
        conn,
    )
    conn.close()
    return df


# ---------------------------------------------------------------- musique
def parse_musique(m, n=5):
    """Résume les n dernières performances de la musique (la plus récente d'abord)."""
    if not isinstance(m, str) or not m:
        return (np.nan,) * 5
    toks = MUSIQUE_TOKEN.findall(m)[:n]
    if not toks:
        return (np.nan,) * 5
    pos = []
    for p, _ in toks:
        if p.isdigit():
            pos.append(10 if p == "0" else int(p))
        else:  # D (disqualifié), A (arrêté), T (tombé), R...
            pos.append(12)
    pos = np.array(pos, dtype=float)
    return (
        pos.mean(),                       # position moyenne récente
        pos[0],                           # dernière position
        (pos == 1).sum(),                 # victoires récentes
        (pos <= 3).sum(),                 # podiums récents
        sum(p.isalpha() for p, _ in toks),  # fautes / disqualifications récentes
    )


# ---------------------------------------------------------- stats cumulées
def add_prior_stats(df, key, prefix, k=10):
    """Taux de victoire / podium d'une entité (cheval, driver, entraîneur) sur ses
    courses ANTÉRIEURES, lissé vers la moyenne globale (k courses fictives)."""
    grp = df.groupby(key, sort=False)
    runs = grp.cumcount()  # nb de courses avant celle-ci
    wins = grp["is_win"].cumsum() - df["is_win"]
    places = grp["is_place"].cumsum() - df["is_place"]
    p_win, p_place = 0.09, 0.27  # a priori ≈ moyennes globales
    df[f"{prefix}_runs"] = runs
    df[f"{prefix}_win_rate"] = (wins + k * p_win) / (runs + k)
    df[f"{prefix}_place_rate"] = (places + k * p_place) / (runs + k)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["jour"] = pd.to_datetime(df["jour"])
    # Clé cheval stable : `id_cheval` n'existe pas dans les données anciennes,
    # on reconstruit son format (NOM-MERE-PERE) pour toutes les années.
    df["horse_key"] = (df["nom"].fillna("") + "-" + df["mere"].fillna("") + "-"
                       + df["pere"].fillna("")).str.upper().str.strip()
    df = df.sort_values(["heure_depart", "course_id", "num_pmu"]).reset_index(drop=True)

    # Cibles (lignes futures sans résultat -> 0, elles n'influencent rien après elles)
    df["is_win"] = (df["ordre_arrivee"] == 1).astype(int)
    df["is_place"] = (df["ordre_arrivee"].between(1, 3)).astype(int)

    # --- Contexte de la course
    df["nb_partants"] = df.groupby("course_id")["num_pmu"].transform("size")
    df["log_prix"] = np.log1p(df["montant_prix"])
    df["recul"] = (df["handicap_distance"] - df["distance"]).where(df["handicap_distance"].notna())

    # --- Carrière du cheval (fournie par le PMU, état avant la course)
    nc = df["nb_courses"].replace(0, np.nan)
    df["career_win_rate"] = df["nb_victoires"] / nc
    df["career_place_rate"] = df["nb_places"] / nc
    df["log_gains"] = np.log1p(df["gains_carriere"].fillna(0))
    df["log_gains_annee"] = np.log1p(df["gains_annee"].fillna(0))
    df["gains_par_course"] = np.log1p(df["gains_carriere"].fillna(0) / nc)

    mus = df["musique"].apply(parse_musique)
    df[["mus_pos_moy", "mus_last", "mus_wins", "mus_podiums", "mus_fautes"]] = pd.DataFrame(
        mus.tolist(), index=df.index
    )

    # --- Marché : cote de référence (disponible avant la course)
    df["p_ref"] = 1 / df["cote_reference"]
    df["p_ref"] = df["p_ref"] / df.groupby("course_id")["p_ref"].transform("sum")

    # --- Historique calculé à partir de notre base
    add_prior_stats(df, "horse_key", "horse")
    add_prior_stats(df, "driver", "driver", k=30)
    add_prior_stats(df, "entraineur", "trainer", k=30)
    prev_day = df.groupby("horse_key")["jour"].shift(1)
    df["jours_repos"] = (df["jour"] - prev_day).dt.days
    # Réduction kilométrique moyenne passée (trot) : plus bas = plus rapide
    rk = df["reduction_km"].where(df["reduction_km"] > 0)
    rk_grp = pd.DataFrame({"s": rk.fillna(0), "n": rk.notna().astype(int)}).groupby(df["horse_key"])
    rk_sum = rk_grp["s"].cumsum() - rk.fillna(0)
    rk_cnt = rk_grp["n"].cumsum() - rk.notna().astype(int)
    df["rk_moy_passee"] = (rk_sum / rk_cnt.replace(0, np.nan))

    # --- Variables relatives à la course (comparaison aux adversaires)
    for col in ["p_ref", "career_win_rate", "log_gains_annee", "driver_win_rate",
                "horse_win_rate", "mus_pos_moy", "rk_moy_passee", "handicap_valeur"]:
        grp = df.groupby("course_id")[col]
        df[f"{col}_rang"] = grp.rank(pct=True)
        df[f"{col}_ecart"] = df[col] - grp.transform("mean")

    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


FEATURES = [
    # contexte
    "specialite", "distance", "nb_partants", "log_prix", "penetrometre", "corde",
    "type_piste", "categorie",
    # cheval
    "age", "sexe", "oeilleres", "deferre", "inedit", "driver_change", "jument_pleine",
    "handicap_valeur", "handicap_poids", "recul", "place_corde",
    "nb_courses", "career_win_rate", "career_place_rate", "log_gains", "log_gains_annee",
    "gains_par_course",
    "mus_pos_moy", "mus_last", "mus_wins", "mus_podiums", "mus_fautes",
    # historique base
    "horse_runs", "horse_win_rate", "horse_place_rate", "jours_repos", "rk_moy_passee",
    "driver_runs", "driver_win_rate", "driver_place_rate",
    "trainer_runs", "trainer_win_rate", "trainer_place_rate",
    # marché
    "p_ref",
] + [f"{c}_{s}" for c in ["p_ref", "career_win_rate", "log_gains_annee", "driver_win_rate",
                          "horse_win_rate", "mus_pos_moy", "rk_moy_passee", "handicap_valeur"]
     for s in ("rang", "ecart")]
