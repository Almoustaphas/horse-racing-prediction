import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time

SITEMAPS = {
    "races": "https://www.paris-turf.com/sitemap/races.xml",
    "horses": "https://www.paris-turf.com/sitemap/horses.xml",
    "meetings": "https://www.paris-turf.com/sitemap/meetings.xml"
}

HEADERS = {
    "User-Agent": "HorseRacingResearch/0.1"
}

session = requests.Session()
session.headers.update(HEADERS)

all_data = []

def inspect_sitemap(name, url):

    print("\n" + "=" * 60)
    print(f"INSPECTION : {name.upper()}")
    print("=" * 60)

    try:
        response = session.get(
            url,
            timeout=30
        )

        print("HTTP Status:", response.status_code)

        response.raise_for_status()

        root = ET.fromstring(response.content)

        namespace = {
            "sm": "http://www.sitemaps.org/schemas/sitemap/0.9"
        }

        # Vérifier le type de sitemap
        print("Structure XML:", root.tag.split("}")[-1])

        # Extraire les URL
        entries = root.findall("sm:url", namespace)

        # Cas où le document contient des sous-sitemaps
        if not entries:
            print("Aucune entrée URL directe.")
            print("Nombre de sous-sitemaps:",
                  len(root.findall("sm:sitemap", namespace)))
            return

        print("Nombre total de pages:", len(entries))

        for entry in entries:

            loc = entry.find("sm:loc", namespace)
            lastmod = entry.find("sm:lastmod", namespace)

            if loc is None:
                continue

            record = {
                "category": name,
                "url": loc.text,
                "lastmod": (
                    lastmod.text if lastmod is not None else None
                )
            }

            all_data.append(record)

        print("\nEXEMPLES D'URL :")

        for record in all_data:
            if record["category"] == name:
                print(record["url"])

                if sum(
                    1 for item in all_data
                    if item["category"] == name
                ) >= 5:
                    break

    except requests.RequestException as error:
        print("Erreur HTTP:", error)

    except ET.ParseError as error:
        print("Erreur XML:", error)


# Inspection séquentielle
for name, url in SITEMAPS.items():

    inspect_sitemap(name, url)

    time.sleep(2)


# Export CSV
if all_data:

    df = pd.DataFrame(all_data)

    df.to_csv(
        "data/raw/sitemap_inventory.csv",
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("INVENTAIRE TERMINÉ")
    print("=" * 60)

    print(df.groupby("category").size())

    print("\nFichier créé : data/raw/sitemap_inventory.csv")

else:
    print("\nAucune URL collectée.")
