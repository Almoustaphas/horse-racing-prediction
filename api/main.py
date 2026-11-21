"""API de pronostics hippiques.

Lancement local :
    uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs

Toutes les routes /v1/* exigent l'en-tête `X-API-Key` (clé créée avec
`python -m api.jobs create-key NOM`).
"""

import os
import threading
import time
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from api import store

PARIS = ZoneInfo("Europe/Paris")
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))

AVERTISSEMENT = (
    "Pronostics statistiques fournis à titre informatif, sans garantie de gain. "
    "Sur les périodes testées, miser sur le premier choix du modèle a été perdant en moyenne. "
    "Les jeux d'argent sont interdits aux mineurs. Jouer comporte des risques : "
    "endettement, dépendance… Appelez le 09 74 75 13 13 (appel non surtaxé)."
)

app = FastAPI(
    title="Pronostics hippiques",
    version="1.0.0",
    description="Probabilités de victoire et de place par partant pour les courses françaises.\n\n"
                + AVERTISSEMENT,
)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_db = store.connect()


def db():
    return _db


# ------------------------------------------------------------ sécurité
class RateLimiter:
    """Limite simple en mémoire : N requêtes par minute et par clé."""

    def __init__(self, per_minute):
        self.per_minute = per_minute
        self.calls = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, who: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.calls[who]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


limiter = RateLimiter(RATE_LIMIT_PER_MIN)


def require_key(key: str | None = Security(api_key_header), conn=Depends(db)) -> str:
    owner = store.check_key(conn, key)
    if owner is None:
        raise HTTPException(401, "Clé d'API absente ou invalide (en-tête X-API-Key).")
    if not limiter.allow(owner):
        raise HTTPException(429, f"Limite de {RATE_LIMIT_PER_MIN} requêtes par minute dépassée.")
    return owner


# ------------------------------------------------------------ schémas
class Partant(BaseModel):
    num_pmu: int
    nom: str
    driver: str | None
    entraineur: str | None
    rang: int
    proba: float
    proba_place: float | None
    proba_top5: float | None
    cote_juste: float
    cote_reference: float | None


class Course(BaseModel):
    course_id: str
    heure_depart: str | None
    hippodrome: str | None
    libelle: str | None
    specialite: str | None
    distance: int | None
    nb_partants: int | None
    quinte: bool
    modele: str
    mis_a_jour: str | None


class CourseAvecPronostics(Course):
    partants: list[Partant]


class Reponse(BaseModel):
    avertissement: str = AVERTISSEMENT


class ReponseCourses(Reponse):
    date: str
    courses: list[Course]


class ReponsePronostics(Reponse):
    date: str
    courses: list[CourseAvecPronostics]


class ReponseCourse(Reponse):
    course: CourseAvecPronostics


class Bilan(BaseModel):
    courses: int
    reussite_premier_choix: float | None
    premier_choix_dans_les_3: float | None
    reussite_favori_marche: float | None
    mises: int
    gains: float
    roi: float | None
    top5_trouves_moyen: float | None
    top5_complet: float | None


class BilanJour(Bilan):
    date: str


class ReponseBilan(Reponse):
    debut: str
    fin: str
    methode: str = ("Dernier pronostic publié avant le départ de chaque course ; mise de 1 € "
                    "au simple gagnant sur le premier choix, payée à la cote PMU finale. "
                    "Courses dont le premier choix est non partant exclues. "
                    "top5 : parmi les 5 chevaux jugés les plus probables d'être dans les 5 premiers, "
                    "combien le sont (sur 5) et part des courses où les 5 le sont (désordre).")
    total: Bilan
    par_jour: list[BilanJour]


# ------------------------------------------------------------ utilitaires
def _heure(ms):
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(PARIS).isoformat(timespec="minutes")


def _course(rows, with_partants=True):
    r0 = rows[0]
    c = {
        "course_id": r0["course_id"], "heure_depart": _heure(r0["heure_depart"]),
        "hippodrome": r0["hippodrome"], "libelle": r0["libelle_course"],
        "specialite": r0["specialite"], "distance": r0["distance"], "nb_partants": r0["nb_partants"],
        "quinte": bool(r0.get("quinte") or 0),
        "modele": r0["modele"], "mis_a_jour": r0["updated_at"],
    }
    if with_partants:
        c["partants"] = [{k: r[k] for k in Partant.model_fields} for r in rows]
    return c


def _group(rows):
    out = {}
    for r in rows:
        out.setdefault(r["course_id"], []).append(r)
    return list(out.values())


def _filtre(rows, quinte: bool | None):
    if quinte is None:
        return rows
    return [r for r in rows if bool(r.get("quinte") or 0) == quinte]


def _jour(d: date | None) -> str:
    return (d or datetime.now(PARIS).date()).isoformat()


# ------------------------------------------------------------ routes
@app.get("/sante", tags=["service"])
def sante(conn=Depends(db)):
    """État du service et dernière mise à jour des pronostics (sans authentification)."""
    return {"statut": "ok", "derniere_mise_a_jour": store.last_refresh(conn)}


@app.get("/v1/courses", response_model=ReponseCourses, tags=["pronostics"])
def courses(date_: date | None = Query(None, alias="date", description="AAAA-MM-JJ (défaut : aujourd'hui)"),
            quinte: bool | None = Query(None, description="true : uniquement les courses support du Quinté+"),
            _owner: str = Depends(require_key), conn=Depends(db)):
    """Liste des courses d'un jour pour lesquelles des pronostics sont disponibles."""
    jour = _jour(date_)
    return {"date": jour, "courses": [_course(g, False) for g in _group(_filtre(store.get_day(conn, jour), quinte))]}


@app.get("/v1/pronostics", response_model=ReponsePronostics, tags=["pronostics"])
def pronostics(date_: date | None = Query(None, alias="date", description="AAAA-MM-JJ (défaut : aujourd'hui)"),
               top: int | None = Query(None, ge=1, le=20, description="ne garder que les N premiers par course"),
               quinte: bool | None = Query(None, description="true : uniquement les courses support du Quinté+"),
               _owner: str = Depends(require_key), conn=Depends(db)):
    """Pronostics de toutes les courses d'un jour, partants classés par probabilité."""
    jour = _jour(date_)
    out = []
    for g in _group(_filtre(store.get_day(conn, jour), quinte)):
        c = _course(g)
        if top:
            c["partants"] = c["partants"][:top]
        out.append(c)
    return {"date": jour, "courses": out}


@app.get("/v1/courses/{course_id}/pronostics", response_model=ReponseCourse, tags=["pronostics"])
def pronostics_course(course_id: str, _owner: str = Depends(require_key), conn=Depends(db)):
    """Pronostics d'une course (identifiant au format AAAA-MM-JJ_R1_C1)."""
    rows = store.get_course(conn, course_id)
    if not rows:
        raise HTTPException(404, f"Aucun pronostic pour la course {course_id}.")
    return {"course": _course(rows)}


@app.get("/v1/bilan", response_model=ReponseBilan, tags=["bilan"])
def bilan(debut: date | None = Query(None, description="AAAA-MM-JJ (défaut : fin - 29 jours)"),
          fin: date | None = Query(None, description="AAAA-MM-JJ (défaut : hier)"),
          quinte: bool | None = Query(None, description="true : uniquement les courses support du Quinté+"),
          _owner: str = Depends(require_key), conn=Depends(db)):
    """Performance réelle des pronostics publiés, comparée aux arrivées officielles."""
    fin = fin or datetime.now(PARIS).date() - timedelta(days=1)
    debut = debut or fin - timedelta(days=29)
    if debut > fin:
        raise HTTPException(422, "debut doit précéder fin.")
    if (fin - debut).days > 366:
        raise HTTPException(422, "Période limitée à un an.")
    df = store.bilan_courses(conn, debut.isoformat(), fin.isoformat())
    if quinte is not None:
        df = df[df["quinte"] == quinte]
    return {"debut": debut.isoformat(), "fin": fin.isoformat(), "total": store.resume_bilan(df),
            "par_jour": [{"date": j, **store.resume_bilan(g)} for j, g in df.groupby("jour")]}
