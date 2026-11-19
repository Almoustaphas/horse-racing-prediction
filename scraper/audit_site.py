
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

BASE_URL = "https://www.paris-turf.com/"
ROBOTS_URL = urljoin(BASE_URL, "robots.txt")

HEADERS = {
    "User-Agent": "HorseRacingResearch/0.1"
}

session = requests.Session()
session.headers.update(HEADERS)

print("=" * 60)
print("AUDIT TECHNIQUE PARIS-TURF")
print("=" * 60)

# 1. Vérifier robots.txt
try:
    robots_response = session.get(
        ROBOTS_URL,
        timeout=20
    )

    print("\nROBOTS.TXT")
    print("HTTP Status:", robots_response.status_code)

    if robots_response.status_code != 200:
        print("Impossible de vérifier les règles robots.txt.")
        raise SystemExit("Audit suspendu.")

    print(robots_response.text[:3000])

    robots = RobotFileParser()
    robots.set_url(ROBOTS_URL)
    robots.parse(robots_response.text.splitlines())

    if not robots.can_fetch(HEADERS["User-Agent"], BASE_URL):
        raise SystemExit(
            "Accès non autorisé par robots.txt. Arrêt."
        )

except requests.RequestException as error:
    raise SystemExit(f"Erreur robots.txt : {error}")

# 2. Examiner la page d'accueil
try:
    response = session.get(
        BASE_URL,
        timeout=30
    )

    print("\nPAGE D'ACCUEIL")
    print("Status:", response.status_code)
    print("URL finale:", response.url)
    print("Content-Type:", response.headers.get("Content-Type"))

    response.raise_for_status()

    if "html" not in response.headers.get("Content-Type", ""):
        raise SystemExit("La réponse n'est pas une page HTML.")

    soup = BeautifulSoup(response.text, "lxml")

    # 3. Informations générales
    print("\nTITRE")
    print(soup.title.get_text(strip=True) if soup.title else "Absent")

    print("\nSTRUCTURE HTML")
    print("Tables:", len(soup.find_all("table")))
    print("Scripts:", len(soup.find_all("script")))
    print("Forms:", len(soup.find_all("form")))

    # 4. Liens internes
    domain = urlparse(BASE_URL).netloc
    links = set()

    for a in soup.find_all("a", href=True):
        link = urljoin(BASE_URL, a["href"])

        if urlparse(link).netloc == domain:
            links.add(link)

    print("\nLIENS INTERNES")
    for link in sorted(links):
        print(link)

    # 5. Enregistrer le HTML pour inspection
    with open(
        "data/raw/homepage.html",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(response.text)

    print("\nHTML sauvegardé dans data/raw/homepage.html")
    print("Audit terminé.")

except requests.RequestException as error:
    print("Erreur HTTP :", error)
