"""Slanje akcija modulima 01 i 02: POST <adresa modula>/<akcija> s parametrima kao JSON."""

import json
import urllib.error
import urllib.request

ADRESE_MODULA = {  # dogovoriti portove
    "hw_adapter": "http://127.0.0.1:8001",
    "windows_agent": "http://127.0.0.1:8002",
}
VREMENSKO_OGRANICENJE = 2


def izvrsi(akcija, probni_rad=True):
    naziv, cilj, parametri = akcija["action"], akcija["target"], akcija["params"]

    if naziv == "NO_ACTION":
        return "preskoceno"

    adresa = f"{ADRESE_MODULA[cilj]}/{naziv}"

    if probni_rad:
        print(f"[PROBNI RAD] POST {adresa} {parametri}")
        return "probni_rad"

    zahtjev = urllib.request.Request(
        adresa,
        data=json.dumps(parametri).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(zahtjev, timeout=VREMENSKO_OGRANICENJE) as odgovor:
            return json.loads(odgovor.read() or b"{}").get("status", "ok")
    except urllib.error.HTTPError as greska:
        print(f"[GREŠKA] {adresa} je vratio {greska.code}")
        return "greska"
    except (urllib.error.URLError, TimeoutError):
        print(f"[NEDOSTUPNO] {cilj} ne odgovara na {adresa}")
        return "nedostupno"
