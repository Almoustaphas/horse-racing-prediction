# horse-racing-prediction

Modèle de prédiction des courses hippiques françaises à partir des données de l'API JSON du PMU.

## Structure

```
scraper/
  pmu_client.py   client HTTP de l'API PMU (limitation de débit, reprises)
  collect.py      collecte historique -> database/pmu.sqlite (reprenable)
  audit_*.py      anciens scripts d'exploration Paris-Turf
prediction/
  features.py     variables par partant (sans fuite d'information)
  train.py        entraînement + évaluation chronologique
  predict.py      pronostics des courses du jour
  experiment.py   comparaison de jeux de variables (walk-forward)
api/
  main.py         API FastAPI (clé d'API, limite de requêtes)
  store.py        base de l'API : pronostics publiés, archive, arrivées, clés, journal
  jobs.py         tâches : publication, collecte quotidienne, bilan, planificateur, clés
tests/            tests (pytest)
```

## Utilisation

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Collecter l'historique (≈ 12 s par jour de courses)
.venv/bin/python -m scraper.collect --start 2023-01-01

# 2. Entraîner et évaluer (test = 6 derniers mois)
.venv/bin/python -m prediction.train                 # gagnant
.venv/bin/python -m prediction.train --target place  # placé
.venv/bin/python -m prediction.train --no-market     # sans les cotes

# Comparer des jeux de variables sur 3 périodes de test successives de 4 mois
.venv/bin/python -m prediction.experiment

# 3. Pronostics du jour
.venv/bin/python -m prediction.predict
```

Si les cotes de référence d'une course ne sont pas encore publiées (souvent le matin),
`predict` bascule automatiquement sur `models/model_win_nomarket.joblib` pour cette course.
Il faut donc avoir lancé les deux entraînements (`train` et `train --no-market`).

Ne lancez pas `train` pendant une collecte : la lecture peut verrouiller la base SQLite
et faire échouer l'écriture d'un jour (il sera repris au lancement suivant de `collect`).

Relancer `scraper.collect` sans `--end` complète la base jusqu'à hier.

## API

L'API sert des pronostics **pré-calculés** stockés dans `database/api.sqlite` ;
elle n'interroge jamais le PMU elle-même.

```bash
.venv/bin/pip install -r requirements-dev.txt

# Publier les pronostics du jour (≈ 2 min), puis créer une clé pour un utilisateur
.venv/bin/python -m api.jobs refresh
.venv/bin/python -m api.jobs create-key alice      # la clé n'est affichée qu'une fois

# Lancer l'API -> documentation interactive sur http://127.0.0.1:8000/docs
.venv/bin/uvicorn api.main:app --reload

# Interroger
curl -H "X-API-Key: <clé>" "http://127.0.0.1:8000/v1/pronostics?top=3"
```

| Route | Description |
|---|---|
| `GET /sante` | état du service et dernière publication (sans clé) |
| `GET /v1/courses?date=AAAA-MM-JJ` | courses du jour ayant des pronostics |
| `GET /v1/pronostics?date=…&top=N` | pronostics de toutes les courses, partants classés |
| `GET /v1/courses/{course_id}/pronostics` | pronostics d'une course (`2026-11-20_R1_C1`) |
| `GET /v1/bilan?debut=…&fin=…` | performance réelle des pronostics archivés (défaut : 30 derniers jours) |

Chaque partant a `proba` (victoire), `proba_place`, `cote_juste` (= 1 / proba),
`cote_reference` (cote PMU si publiée) et `rang` ; chaque course indique `modele`
(« avec cotes » ou « sans cotes »). Toutes les réponses portent un avertissement jeu responsable.

### Archive et bilan

À chaque publication, les pronostics des courses **pas encore parties** sont copiés dans
la table `pronostics_archive` ; une fois la course partie, sa version n'est plus modifiée.
Chaque matin, `daily` copie les arrivées de la veille (table `resultats`) et le bilan compare :
réussite du premier choix, premier choix dans les 3, réussite du favori du marché,
et ROI d'une mise de 1 € au simple gagnant sur le premier choix (cote PMU finale).
Les courses dont le premier choix est non partant sont exclues.

```bash
.venv/bin/python -m api.jobs resultats --date 2026-11-20   # après la collecte de ce jour
.venv/bin/python -m api.jobs bilan --debut 2026-11-01
```

Gestion des clés : `api.jobs list-keys`, `api.jobs revoke-key NOM`. Les clés sont stockées hachées (SHA-256).
Limite : `RATE_LIMIT_PER_MIN` requêtes par minute et par utilisateur (60 par défaut).

### Docker

```bash
docker compose up -d --build
docker compose exec api python -m api.jobs create-key alice
```

- `api` : le serveur, port 8000.
- `scheduler` : collecte la veille, calcule son bilan, publie les pronostics chaque jour à 6 h, puis les recalcule
  toutes les 15 min de 7 h à 23 h (heure de Paris), à mesure que les cotes sont publiées.

`database/` et `models/` sont montés depuis la machine hôte : il faut y avoir la base collectée
et les trois modèles entraînés. En production, placer l'API derrière un proxy HTTPS (Caddy, Nginx).

- Mémoire : le calcul des pronostics charge tout l'historique et culmine vers **2 Go**.
  Prévoir au moins 3 Go pour Docker (Docker Desktop : Paramètres → Resources → Memory).
- Docker Desktop fait tourner les conteneurs dans une machine virtuelle : les verrous SQLite
  ne sont pas partagés avec la machine hôte. Ne lancez pas de commande sur `database/`
  depuis l'hôte pendant que les conteneurs tournent (sans risque avec Docker Engine sous Linux).

```bash
.venv/bin/pytest      # tests
```

## Méthode

- **Données** : courses en France ayant une arrivée. Une ligne par partant :
  carrière, musique, driver/jockey, entraîneur, poids/recul, corde, terrain, cote de référence.
- **Variables historiques** : taux de victoire et de podium du cheval, du driver et de l'entraîneur,
  calculés uniquement sur les courses antérieures et lissés vers la moyenne globale.
  S'y ajoutent les jours de repos, la réduction kilométrique passée et la position de chaque partant
  par rapport aux autres de la course.
- **Modèle** : `HistGradientBoostingClassifier` ; les probabilités sont normalisées pour sommer à 1 dans chaque course.
- **Évaluation** : découpage dans le temps, comparée au marché (cotes) et au hasard.
  Mesures : log-loss, réussite du premier choix, ROI au simple gagnant.

⚠️ `cote_direct` est la dernière cote avant le départ ; elle sert uniquement à calculer le ROI.
`ordre_arrivee`, `incident`, `temps_obtenu`, `reduction_km` et `ecart` sont des résultats :
ne jamais les utiliser comme variables de la course en cours.
