"""Prépare l'archive des données à envoyer sur le disque de production.

    python -m deploy.pack_data            # -> export/donnees.tar.gz

Contient une copie cohérente de database/pmu.sqlite (sauvegarde SQLite, possible pendant
une collecte) et les modèles. La base de l'API (clés, archive) n'est pas incluse :
la production démarre avec la sienne.
Sur le serveur : tar -xzf donnees.tar.gz -C /app/data, puis redémarrer le service.
"""

import sqlite3
import tarfile
import tempfile
from pathlib import Path

from prediction.features import DB_PATH
from prediction.predict import MODELS_DIR

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["model_win.joblib", "model_win_nomarket.joblib", "model_place.joblib"]


def main():
    missing = [m for m in MODELS if not (MODELS_DIR / m).exists()]
    if missing:
        raise SystemExit(f"Modèles manquants : {missing} (lancer prediction.train)")
    out = ROOT / "export" / "donnees.tar.gz"
    out.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "pmu.sqlite"
        src, dst = sqlite3.connect(DB_PATH), sqlite3.connect(copy)
        with dst:
            src.backup(dst)
        src.close(), dst.close()
        with tarfile.open(out, "w:gz") as tar:
            tar.add(copy, arcname="pmu.sqlite")
            for m in MODELS:
                tar.add(MODELS_DIR / m, arcname=f"models/{m}")
    print(f"{out} ({out.stat().st_size / 1e6:.0f} Mo)")


if __name__ == "__main__":
    main()
