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
