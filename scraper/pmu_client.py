"""Client minimal pour l'API JSON (non documentée) du PMU.

Endpoints utilisés :
    /programme/{JJMMAAAA}                       -> réunions + courses du jour
    /programme/{JJMMAAAA}/R{n}/C{m}/participants -> partants d'une course
"""

import threading
import time
from datetime import date

import requests

BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1/programme"
HEADERS = {"User-Agent": "HorseRacingResearch/0.1", "Accept": "application/json"}


class PMUClient:
    def __init__(self, min_interval=0.25, timeout=30, max_retries=4):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.min_interval = min_interval  # délai minimal entre deux requêtes (toutes threads)
        self.timeout = timeout
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._last_call = 0.0

    def _throttle(self):
        with self._lock:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _get(self, url):
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(
                    url, params={"specialisation": "OFFLINE"}, timeout=self.timeout
                )
                if resp.status_code in (204, 404):
                    return None
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt * 2)
        return None

    @staticmethod
    def _fmt(day: date) -> str:
        return day.strftime("%d%m%Y")

    def programme(self, day: date):
        data = self._get(f"{BASE_URL}/{self._fmt(day)}")
        return data.get("programme") if data else None

    def participants(self, day: date, reunion: int, course: int):
        data = self._get(f"{BASE_URL}/{self._fmt(day)}/R{reunion}/C{course}/participants")
        return data.get("participants", []) if data else []
