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
deploy/           production : démarrage, raccourci `hrp`, archive des données
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
.venv/bin/python -m prediction.train --target top5   # dans les 5 premiers
# Versions sans cotes des modèles placé et top 5 (repli le matin, voir plus bas)
.venv/bin/python -m prediction.train --target place --no-market
.venv/bin/python -m prediction.train --target top5 --no-market

# Comparer des jeux de variables sur 3 périodes de test successives de 4 mois
.venv/bin/python -m prediction.experiment

# 3. Pronostics du jour
.venv/bin/python -m prediction.predict
```

Si les cotes de référence d'une course ne sont pas encore publiées (souvent le matin),
`predict` bascule automatiquement sur `models/model_win_nomarket.joblib` pour cette course.
Il faut donc avoir lancé au moins `train`, `train --no-market` et `train --target place` ;
les modèles top 5 et les versions sans cotes du placé et du top 5 sont utilisés s'ils existent.

Ne lancez pas `train` pendant une collecte : la lecture peut verrouiller la base SQLite
et faire échouer l'écriture d'un jour (il sera repris au lancement suivant de `collect`).

Relancer `scraper.collect` sans `--end` complète la base jusqu'à hier.
La colonne `courses.quinte` repère la course support du Quinté+ (une par jour) ;
`scraper.collect --backfill-quinte` la renseigne pour une base collectée avant son ajout.

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
| `GET /v1/pronostics?date=…&top=N&quinte=true` | pronostics de toutes les courses (ou du seul Quinté+), partants classés |
| `GET /v1/courses/{course_id}/pronostics` | pronostics d'une course (`2026-11-20_R1_C1`) |
| `GET /v1/bilan?debut=…&fin=…&quinte=true` | performance réelle des pronostics archivés (défaut : 30 derniers jours) |

Chaque partant a `proba` (victoire), `proba_place` (dans les 3 premiers), `proba_top5`
(dans les 5 premiers), `cote_juste` (= 1 / proba), `cote_reference` (cote PMU si publiée) et `rang` ;
dans une course, ces probabilités somment à 1, 3 et 5. Chaque course indique `modele`
(« avec cotes » ou « sans cotes ») et `quinte` (support du Quinté+). Toutes les réponses portent un avertissement jeu responsable.

### Archive et bilan

À chaque publication, les pronostics des courses **pas encore parties** sont copiés dans
la table `pronostics_archive` ; une fois la course partie, sa version n'est plus modifiée.
Chaque matin, `daily` copie les arrivées de la veille (table `resultats`) et le bilan compare :
réussite du premier choix, premier choix dans les 3, réussite du favori du marché,
ROI d'une mise de 1 € au simple gagnant sur le premier choix (cote PMU finale), et nombre
de chevaux trouvés parmi les 5 premiers (les 5 jugés les plus probables d'y être).
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

- Mémoire : le calcul des pronostics charge tout l'historique et culmine vers **1,5 Go**.
  Prévoir au moins 2 Go pour Docker (Docker Desktop : Paramètres → Resources → Memory).
- Docker Desktop fait tourner les conteneurs dans une machine virtuelle : les verrous SQLite
  ne sont pas partagés avec la machine hôte. Ne lancez pas de commande sur `database/`
  depuis l'hôte pendant que les conteneurs tournent (sans risque avec Docker Engine sous Linux).

```bash
.venv/bin/pytest      # tests
```

### Déploiement sur Render

`render.yaml` décrit un seul service web Docker (offre *Standard* : 1 CPU, 2 Go, région Francfort)
qui fait tourner l'API **et** le planificateur, avec un disque persistant de 2 Go monté sur
`/app/data` (un disque Render n'est attaché qu'à un service ; l'offre gratuite n'en a pas).
Les déploiements avec disque ne sont pas sans coupure : quelques secondes d'arrêt à chaque mise à jour.

1. **Créer le service** : sur dashboard.render.com, *New → Blueprint*, choisir le dépôt GitHub.
   Render lit `render.yaml` et crée le service et son disque (payant : voir les tarifs affichés).
   Chaque `git push` sur `main` redéploie.
2. **Envoyer les données** (le disque est vide au départ ; le service répond mais attend les données) :
   ```bash
   .venv/bin/python -m deploy.pack_data    # -> export/donnees.tar.gz (≈ 70 Mo : base + modèles)
   ```
   Ajouter sa clé SSH publique dans *Account Settings → SSH Public Keys*, puis copier l'adresse
   indiquée dans l'onglet *Connect → SSH* du service :
   ```bash
   scp -s export/donnees.tar.gz <adresse-ssh-du-service>:/tmp/
   ```
   Dans l'onglet *Shell* du service (ou via `ssh`) :
   ```bash
   tar -xzf /tmp/donnees.tar.gz -C /app/data && rm /tmp/donnees.tar.gz
   ```
   Le planificateur détecte les données dans la minute : collecte de la veille, puis pronostics.
3. **Créer les clés** dans le *Shell* du service : `hrp create-key alice` (aussi `hrp list-keys`,
   `hrp revoke-key`, `hrp bilan`). Toujours passer par `hrp` : il s'exécute sous l'utilisateur du
   service (une commande lancée en root créerait des fichiers SQLite illisibles par l'API).
4. **Vérifier** : `https://<nom-du-service>.onrender.com/sante` puis `/docs`.

Mettre à jour les modèles après un réentraînement local : refaire l'étape 2 (la base de production,
alimentée chaque jour par le planificateur, est alors remplacée par la copie locale).
Les journaux du service (*Logs*) montrent chaque collecte, bilan et publication.

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
