import requests
import xml.etree.ElementTree as ET

URL = "https://www.paris-turf.com/sitemap.xml"

try:
    response = requests.get(
        URL,
        timeout=30,
        headers={
            "User-Agent": "HorseRacingResearch/0.1"
        }
    )

    print("HTTP Status:", response.status_code)

    response.raise_for_status()

    root = ET.fromstring(response.content)

    print("\nRACINES XML :", root.tag)

    urls = []

    for element in root.iter():
        if element.tag.endswith("loc"):
            urls.append(element.text)

    print("\nNOMBRE D'URL :", len(urls))

    print("\nPREMIERES URL :")

    for url in urls[:30]:
        print(url)

except requests.RequestException as error:
    print("Erreur HTTP :", error)

except ET.ParseError as error:
    print("Erreur XML :", error)
