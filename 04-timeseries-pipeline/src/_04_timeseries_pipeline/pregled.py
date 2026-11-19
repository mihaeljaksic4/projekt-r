#isto ai, nije moje
import pandas
from pathlib import Path

MAPA = Path("pregledi")
MAPA.mkdir(exist_ok=True)


def spremi_pregled(naziv, podaci):
    putanja = MAPA / f"{naziv}.csv"

    if isinstance(podaci, pandas.DataFrame):
        tablica = podaci.copy()

    elif isinstance(podaci, dict):
        tablica = pandas.DataFrame([podaci])

    else:
        tablica = pandas.DataFrame(podaci)

    if tablica.empty:
        print(f"[PREGLED] {naziv}: nema podataka.")
        return

    # Ako datoteka već postoji, nadopuni je.
    # Ako ne postoji, stvori novu i zapiši zaglavlje.
    postoji = putanja.exists() and putanja.stat().st_size > 0

    tablica.to_csv(
        putanja,
        mode="a",
        header=not postoji,
        index=True,
        encoding="utf-8-sig"
    )

    print(
        f"[PREGLED] {naziv}: "
        f"dodano {len(tablica)} redaka | "
        f"ukupna datoteka: {putanja}"
    )
