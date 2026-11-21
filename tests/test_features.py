"""Vérifie que les statistiques historiques n'utilisent que le passé."""

import pandas as pd
import pytest

from prediction.features import add_prior_stats, add_window_stats, parse_musique


def runs(rows):
    df = pd.DataFrame(rows, columns=["jour", "driver", "is_win"])
    df["jour"] = pd.to_datetime(df["jour"])
    df["is_place"] = df["is_win"]
    return df


def test_prior_stats_exclut_la_course_courante():
    df = runs([("2026-01-01", "A", 1), ("2026-01-02", "A", 1), ("2026-01-03", "A", 0)])
    add_prior_stats(df, "driver", "d", k=0.000001)
    assert df["d_runs"].tolist() == [0, 1, 2]
    assert df["d_win_rate"].iloc[1] == pytest.approx(1.0)   # seule la 1re course est vue
    assert df["d_win_rate"].iloc[2] == pytest.approx(1.0)   # 2 victoires sur 2, pas la 3e


def test_fenetre_90_jours_exclut_le_jour_meme_et_le_vieux():
    df = runs([("2026-01-01", "A", 1), ("2026-05-01", "A", 1),
               ("2026-05-10", "A", 0), ("2026-05-10", "A", 1)])
    add_window_stats(df, "driver", "d", days=90)
    # Le 01/01 est hors fenêtre ; le jour même (10/05) n'est pas compté
    assert df["d_runs_90j"].tolist() == [0, 0, 1, 1]


def test_parse_musique():
    pos_moy, last, wins, podiums, fautes = parse_musique("1a3aDa0a(25)2a")
    assert last == 1 and wins == 1 and podiums == 3 and fautes == 1
    assert pos_moy == pytest.approx((1 + 3 + 12 + 10 + 2) / 5)


def test_is_quinte():
    from scraper.collect import is_quinte
    assert is_quinte({"paris": [{"typePari": "TIERCE"}, {"typePari": "QUINTE_PLUS"}]}) == 1
    assert is_quinte({"paris": [{"typePari": "SUPER_PICK5"}]}) == 0
    assert is_quinte({}) == 0


def test_evaluate_top5():
    from prediction.train import evaluate
    # 6 partants ; le modèle préfère 1 à 6 ; les 5 premiers réels sont 1, 2, 3, 4, 6
    d = pd.DataFrame({"course_id": ["c"] * 6, "p": [6, 5, 4, 3, 2, 1],
                      "is_top5": [1, 1, 1, 1, 0, 1], "cote_direct": [None] * 6})
    out = evaluate(d, "p", "is_top5")
    assert out["top5_trouves"] == 4 and out["top5_complet"] == 0
