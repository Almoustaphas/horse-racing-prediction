"""Construction des variables (features) par partant.

Règle d'or : chaque variable ne doit utiliser QUE de l'information disponible
avant le départ de la course. Les statistiques historiques (cheval, driver,
entraîneur) sont calculées en cumul sur les courses antérieures uniquement
(tri chronologique + exclusion de la ligne courante).
"""

import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(os.environ.get("PMU_DB_PATH", Path(__file__).resolve().parent.parent / "database" / "pmu.sqlite"))

MUSIQUE_TOKEN = re.compile(r"(\d|[A-Z])([a-z])")

# Variables catégorielles encodées par le modèle
CATEGORICAL = ["specialite", "sexe", "oeilleres", "deferre", "corde", "type_piste", "categorie"]


# Colonnes de la base jamais utilisées par le modèle : non chargées (mémoire)
UNUSED_COLUMNS = {
    "id_cheval", "race", "pays_cheval", "proprietaire", "eleveur", "pere_mere",
    "nb_places_2", "nb_places_3", "gains_victoires", "gains_place", "gains_annee_prec",
    "supplement", "favori", "temps_obtenu",
}


def load_raw(db_path=DB_PATH, chunksize=50_000) -> pd.DataFrame:
    """Charge les partants avec leur course. Lecture par tranches pour limiter le pic
    de mémoire (≈ 2 Go en une fois, la machine de production n'en a pas autant)."""
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(participants)") if r[1] not in UNUSED_COLUMNS]
    has_quinte = "quinte" in {r[1] for r in conn.execute("PRAGMA table_info(courses)")}
    chunks = pd.read_sql_query(
        f"""
        SELECT {", ".join("p." + c for c in cols)},
               c.jour, c.heure_depart, c.hippodrome, c.discipline, c.specialite,
               c.distance, c.corde, c.type_piste, c.categorie, c.montant_prix,
               c.penetrometre, c.temperature{", c.quinte" if has_quinte else ""}
        FROM participants p JOIN courses c USING (course_id)
        WHERE p.statut = 'PARTANT'
        """,
        conn, chunksize=chunksize,
    )
    df = pd.concat(list(chunks), ignore_index=True)
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


def add_window_stats(df, key, prefix, days=90, k=20):
    """Forme récente : taux de victoire / podium sur les `days` jours précédant le
    jour de la course (le jour même est exclu)."""
    # Une ligne par (entité, jour), triée par entité puis par jour
    daily = (df.groupby([key, "jour"])
               .agg(w=("is_win", "sum"), p=("is_place", "sum"), n=("is_win", "size"))
               .reset_index().set_index("jour"))
    # Somme glissante sur [jour - days, jour[ ; résultat indexé (entité, jour), unique
    daily = (daily.groupby(key)[["w", "p", "n"]]
                  .rolling(f"{days}D", closed="left").sum().fillna(0).reset_index())
    daily[f"{prefix}_runs_{days}j"] = daily["n"]
    daily[f"{prefix}_win_{days}j"] = (daily["w"] + k * 0.09) / (daily["n"] + k)
    daily[f"{prefix}_place_{days}j"] = (daily["p"] + k * 0.27) / (daily["n"] + k)
    cols = [f"{prefix}_runs_{days}j", f"{prefix}_win_{days}j", f"{prefix}_place_{days}j"]
    merged = df[[key, "jour"]].merge(daily[[key, "jour"] + cols], on=[key, "jour"], how="left")
    df[cols] = merged[cols].to_numpy()


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
    df["is_top5"] = (df["ordre_arrivee"].between(1, 5)).astype(int)

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
    df["p_ref"] = 1 / df["cote_reference"].where(df["cote_reference"] > 1)  # cote 0 = absente
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

    # --- Forme récente (fenêtre glissante de 90 jours)
    add_window_stats(df, "driver", "driver")
    add_window_stats(df, "entraineur", "trainer")

    # --- Aptitudes du cheval et associations (historique de la base)
    df["dist_band"] = df["specialite"].astype(str) + "_" + (df["distance"] // 400).astype(str)
    add_prior_stats(df, ["horse_key", "dist_band"], "horse_dist", k=5)
    add_prior_stats(df, ["horse_key", "hippodrome"], "horse_hippo", k=5)
    add_prior_stats(df, ["horse_key", "driver"], "duo", k=5)
    add_prior_stats(df, ["driver", "hippodrome"], "driver_hippo", k=20)

    # --- Dernière course du cheval dans la base
    hg = df.groupby("horse_key")
    df["prev_driver_diff"] = (hg["driver"].shift(1) != df["driver"]).astype(float).where(
        hg["driver"].shift(1).notna())
    df["prev_pos"] = hg["ordre_arrivee"].shift(1).fillna(12).where(hg.cumcount() > 0)
    df["prev_p_ref"] = hg["p_ref"].shift(1)
    df["prev_nb_partants"] = hg["nb_partants"].shift(1)
    df["delta_prix"] = df["log_prix"] - hg["log_prix"].shift(1)       # monte / descend de catégorie
    df["delta_distance"] = df["distance"] - hg["distance"].shift(1)

    # --- Variables relatives à la course (comparaison aux adversaires)
    for col in ["p_ref", "career_win_rate", "log_gains_annee", "driver_win_rate",
                "horse_win_rate", "mus_pos_moy", "rk_moy_passee", "handicap_valeur",
                "driver_win_90j", "trainer_win_90j", "horse_dist_win_rate", "prev_p_ref"]:
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

# Variables ajoutées dans la v2 (forme récente, aptitudes, dernière course)
NEW_FEATURES = [
    "driver_runs_90j", "driver_win_90j", "driver_place_90j",
    "trainer_runs_90j", "trainer_win_90j", "trainer_place_90j",
    "horse_dist_runs", "horse_dist_win_rate", "horse_dist_place_rate",
    "horse_hippo_runs", "horse_hippo_win_rate", "horse_hippo_place_rate",
    "duo_runs", "duo_win_rate", "duo_place_rate",
    "driver_hippo_win_rate", "driver_hippo_place_rate",
    "prev_driver_diff", "prev_pos", "prev_p_ref", "prev_nb_partants",
    "delta_prix", "delta_distance",
] + [f"{c}_{s}" for c in ["driver_win_90j", "trainer_win_90j", "horse_dist_win_rate", "prev_p_ref"]
     for s in ("rang", "ecart")]

FEATURES_V1 = list(FEATURES)
FEATURES = FEATURES_V1 + NEW_FEATURES
