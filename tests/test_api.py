import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main, store

DAY = "2026-11-20"


def fake_predictions():
    rows = []
    for c, n in (("C1", 3), ("C2", 2)):
        probs = [0.5, 0.3, 0.2][:n]
        for i, p in enumerate(probs, start=1):
            rows.append({
                "course_id": f"{DAY}_R1_{c}", "jour": DAY,
                "heure_depart": 1795640400000 + (0 if c == "C1" else 1_800_000),
                "hippodrome": "VIN", "libelle_course": f"PRIX {c}", "specialite": "TROT_ATTELE",
                "distance": 2700, "nb_partants": n, "num_pmu": i, "nom": f"CHEVAL {c}{i}",
                "driver": "X. DRIVER", "entraineur": "Y. ENTRAINEUR",
                "proba": p, "proba_place": min(0.99, p * 2), "cote_juste": 1 / p,
                "cote_reference": None if c == "C2" else 1 / p + 1,
                "rang": i, "modele": "avec cotes",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "api.sqlite")
    store.save_predictions(c, DAY, fake_predictions())
    store.log_refresh(c, DAY, 2, "ok")
    return c


@pytest.fixture
def client(conn):
    main.app.dependency_overrides[main.db] = lambda: conn
    main.limiter.calls.clear()
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


@pytest.fixture
def key(conn):
    return store.create_key(conn, "testeur")


def test_sante_sans_cle(client):
    r = client.get("/sante")
    assert r.status_code == 200
    assert r.json()["derniere_mise_a_jour"]["statut"] == "ok"


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "hrp_mauvaise"}])
def test_cle_obligatoire(client, headers):
    assert client.get("/v1/pronostics", params={"date": DAY}, headers=headers).status_code == 401


def test_cle_stockee_hachee(conn, key):
    stored = [r[0] for r in conn.execute("SELECT key_hash FROM api_keys")]
    assert key not in stored and len(stored[0]) == 64


def test_liste_courses(client, key):
    r = client.get("/v1/courses", params={"date": DAY}, headers={"X-API-Key": key})
    assert r.status_code == 200
    body = r.json()
    assert [c["course_id"] for c in body["courses"]] == [f"{DAY}_R1_C1", f"{DAY}_R1_C2"]
    assert body["courses"][0]["heure_depart"].endswith("+01:00")  # heure de Paris
    assert "mineurs" in body["avertissement"]


def test_pronostics_tries_et_top(client, key):
    r = client.get("/v1/pronostics", params={"date": DAY, "top": 2}, headers={"X-API-Key": key})
    c1 = r.json()["courses"][0]
    assert [p["rang"] for p in c1["partants"]] == [1, 2]
    assert c1["partants"][0]["proba"] == pytest.approx(0.5)


def test_pronostics_course_et_404(client, key):
    h = {"X-API-Key": key}
    r = client.get(f"/v1/courses/{DAY}_R1_C2/pronostics", headers=h)
    assert r.status_code == 200
    assert r.json()["course"]["partants"][0]["cote_reference"] is None
    assert client.get(f"/v1/courses/{DAY}_R9_C9/pronostics", headers=h).status_code == 404


def test_jour_sans_pronostic(client, key):
    r = client.get("/v1/pronostics", params={"date": "2020-01-01"}, headers={"X-API-Key": key})
    assert r.status_code == 200 and r.json()["courses"] == []


def test_limite_de_requetes(client, key, monkeypatch):
    monkeypatch.setattr(main.limiter, "per_minute", 2)
    h = {"X-API-Key": key}
    codes = [client.get("/v1/courses", params={"date": DAY}, headers=h).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_cle_revoquee(client, conn, key):
    store.revoke_owner(conn, "testeur")
    assert client.get("/v1/courses", headers={"X-API-Key": key}).status_code == 401


def test_publication_remplace_le_jour(conn):
    store.save_predictions(conn, DAY, fake_predictions().head(2))
    assert len(store.get_day(conn, DAY)) == 2


# ------------------------------------------------------------- archive et bilan
C1_DEPART = 1795640400000


def archive(conn, course_id):
    return {r["num_pmu"]: r["proba"] for r in conn.execute(
        "SELECT num_pmu, proba FROM pronostics_archive WHERE course_id = ?", (course_id,))}


def test_archive_figee_au_depart(conn):
    df = fake_predictions()
    df["proba"] = 0.1
    # Nouvelle publication après le départ de C1 mais avant celui de C2
    store.save_predictions(conn, DAY, df, now_ms=C1_DEPART + 60_000)
    assert set(archive(conn, f"{DAY}_R1_C1").values()) == {0.5, 0.3, 0.2}  # inchangée
    assert set(archive(conn, f"{DAY}_R1_C2").values()) == {0.1}            # mise à jour
    assert {r["proba"] for r in store.get_day(conn, DAY)} == {0.1}           # publication courante


def test_archive_retire_un_non_partant(conn):
    store.save_predictions(conn, DAY, fake_predictions().query("num_pmu != 3"), now_ms=0)
    assert set(archive(conn, f"{DAY}_R1_C1")) == {1, 2}


def test_reprise_des_pronostics_anterieurs(tmp_path):
    c = store.connect(tmp_path / "a.sqlite")
    store.save_predictions(c, DAY, fake_predictions())
    c.execute("DELETE FROM pronostics_archive")
    c.commit()
    c2 = store.connect(tmp_path / "a.sqlite")
    assert c2.execute("SELECT COUNT(*) FROM pronostics_archive").fetchone()[0] == 5


def results():
    # C1 : le premier choix (n°1) gagne à 3,0 ; C2 : le premier choix finit 2e
    return pd.DataFrame([
        (f"{DAY}_R1_C1", 1, "PARTANT", 1, 3.0), (f"{DAY}_R1_C1", 2, "PARTANT", 2, 4.0),
        (f"{DAY}_R1_C1", 3, "PARTANT", 3, 6.0),
        (f"{DAY}_R1_C2", 1, "PARTANT", 2, 2.5), (f"{DAY}_R1_C2", 2, "PARTANT", 1, 3.5),
    ], columns=["course_id", "num_pmu", "statut", "ordre_arrivee", "cote_direct"])


def test_bilan(conn):
    store.save_results(conn, DAY, results())
    r = store.resume_bilan(store.bilan_courses(conn, DAY, DAY))
    assert r["courses"] == 2 and r["reussite_premier_choix"] == 0.5
    assert r["premier_choix_dans_les_3"] == 1.0
    assert r["mises"] == 2 and r["gains"] == 3.0 and r["roi"] == pytest.approx(0.5)
    assert r["reussite_favori_marche"] == 1.0  # C2 n'a pas de cote de référence


def test_bilan_premier_choix_non_partant_exclu(conn):
    res = results()
    res.loc[0, ["statut", "ordre_arrivee"]] = ["NON_PARTANT", None]
    res.loc[1, "ordre_arrivee"] = 1
    store.save_results(conn, DAY, res)
    assert store.resume_bilan(store.bilan_courses(conn, DAY, DAY))["courses"] == 1


def test_route_bilan(client, conn, key):
    store.save_results(conn, DAY, results())
    h = {"X-API-Key": key}
    r = client.get("/v1/bilan", params={"debut": DAY, "fin": DAY}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["total"]["roi"] == pytest.approx(0.5)
    assert [j["date"] for j in body["par_jour"]] == [DAY]
    assert client.get("/v1/bilan", params={"debut": "2026-12-01", "fin": DAY}, headers=h).status_code == 422
    assert client.get("/v1/bilan").status_code == 401


# ------------------------------------------------------------- top 5 et Quinté+
def big_course(n=7, quinte=1):
    """Une course de n partants : le modèle classe 1..n, proba_top5 décroissante."""
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "course_id": f"{DAY}_R1_C9", "jour": DAY, "heure_depart": C1_DEPART + 3_600_000,
            "hippodrome": "VIN", "libelle_course": "PRIX DU QUINTE", "specialite": "PLAT",
            "distance": 2000, "nb_partants": n, "num_pmu": i, "nom": f"Q{i}", "driver": "D",
            "entraineur": "E", "proba": (n + 1 - i) / 28, "proba_place": None,
            "proba_top5": min(0.99, (n + 1 - i) / 6), "cote_juste": 28 / (n + 1 - i),
            "cote_reference": 28 / (n + 1 - i), "rang": i, "modele": "avec cotes", "quinte": quinte,
        })
    return pd.DataFrame(rows)


def test_filtre_quinte(client, conn, key):
    store.save_predictions(conn, DAY, pd.concat([fake_predictions(), big_course()]))
    h = {"X-API-Key": key}
    r = client.get("/v1/pronostics", params={"date": DAY, "quinte": "true"}, headers=h).json()
    assert [c["course_id"] for c in r["courses"]] == [f"{DAY}_R1_C9"]
    assert r["courses"][0]["quinte"] is True
    assert r["courses"][0]["partants"][0]["proba_top5"] == pytest.approx(0.99)
    tous = client.get("/v1/courses", params={"date": DAY}, headers=h).json()["courses"]
    assert len(tous) == 3 and [c["quinte"] for c in tous][:2] == [False, False]


def test_bilan_top5(conn):
    store.save_predictions(conn, DAY, big_course(), now_ms=0)
    # Arrivée : 1, 2, 3, 6, 4 -> 4 des 5 premiers choix (1 à 5) sont dans les 5 premiers
    ordre = {1: 1, 2: 2, 3: 3, 6: 4, 4: 5, 5: 6, 7: 7}
    res = pd.DataFrame([(f"{DAY}_R1_C9", n, "PARTANT", o, 3.0) for n, o in ordre.items()],
                       columns=["course_id", "num_pmu", "statut", "ordre_arrivee", "cote_direct"])
    store.save_results(conn, DAY, res)
    df = store.bilan_courses(conn, DAY, DAY)
    r = store.resume_bilan(df[df["quinte"]])
    assert r["courses"] == 1 and r["top5_trouves_moyen"] == 4 and r["top5_complet"] == 0.0


def test_ancienne_base_mise_a_niveau(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE pronostics (course_id TEXT, num_pmu INTEGER, jour TEXT, heure_depart INTEGER, "
                "hippodrome TEXT, libelle_course TEXT, specialite TEXT, distance INTEGER, nb_partants INTEGER, "
                "nom TEXT, driver TEXT, entraineur TEXT, proba REAL, proba_place REAL, cote_juste REAL, "
                "cote_reference REAL, rang INTEGER, modele TEXT, updated_at TEXT, PRIMARY KEY (course_id, num_pmu))")
    old.commit(); old.close()
    c = store.connect(path)
    cols = {r[1] for r in c.execute("PRAGMA table_info(pronostics)")}
    assert {"proba_top5", "quinte"} <= cols
    store.save_predictions(c, DAY, big_course())
    assert len(store.get_day(c, DAY)) == 7
