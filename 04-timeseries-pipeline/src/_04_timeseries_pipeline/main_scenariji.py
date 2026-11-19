import contextlib
import importlib
import io
from pathlib import Path

import pandas

from . import pregled, state_model_interface
from .generator import generiraj_uzorke, generiraj_evente, SCENARIJI
from . import UtijekuObrada          # klasa je u __init__.py
from .obrada import obradi
from .pipeline import napravi_za_testing

#pokretanje
#scenariji = "_04_timeseries_pipeline.main_scenariji:main" - deni v pyproject.toml
#uv run scenariji
#ovo sve za podatke scenariji i generator je ai, nije moje :)

TRAJANJE = 400            # sekundi po sesiji
SESIJA_RAZMAK = 7200.0    # sesije se vremenski ne preklapaju


def jedna_sesija(scenarij, seed, pocetak):
    uzorci = generiraj_uzorke(trajanje=TRAJANJE, pocetak=pocetak, seed=seed, scenarij=scenarij)
    eventi = generiraj_evente(pocetak, seed=seed, scenarij=scenarij, trajanje=TRAJANJE)

    # NOVI State Model za svaku sesiju (on pamti stanje, ne smije nositi staro u novu sesiju)
    importlib.reload(state_model_interface)

    # pregledi svake sesije idu u svoju mapu: pregledi/<scenarij>_<seed>/
    pregled.MAPA = Path("pregledi") / f"{scenarij}_{seed}"
    pregled.MAPA.mkdir(parents=True, exist_ok=True)

    obrada = UtijekuObrada()
    iu = ie = 0
    with contextlib.redirect_stdout(io.StringIO()):          # tvoj korak() puno ispisuje
        for s in range(TRAJANJE):
            sada = pocetak + s + 1
            while iu < len(uzorci) and uzorci[iu]["timestamp"] < sada:
                obrada.novi_uzorak(uzorci[iu]); iu += 1
            while ie < len(eventi) and eventi[ie]["timestamp"] < sada:
                obrada.novi_event(eventi[ie]); ie += 1
            obrada.korak(sada)

    w = obradi(obrada.uzorci, obrada.eventi)
    return napravi_za_testing(w, obrada.redci, session_id=f"{scenarij}_{seed}"), w


def main(seedovi=(1, 2)):
    skupovi, k = [], 0
    for scenarij in SCENARIJI:
        for seed in seedovi:
            skup, w = jedna_sesija(scenarij, seed, 1730000000.0 + k * SESIJA_RAZMAK)
            k += 1
            print(f"{scenarij:12} seed {seed}: rupa {int(w['rupa'].sum()):3d} s, redaka u datasetu {len(skup):3d}")
            skupovi.append(skup)

    skup = pandas.concat(skupovi)
    print("\nUkupno:", skup.shape)
    print(skup["current_state"].value_counts().to_string())
    skup.to_csv("rezultati_sve_sesije.csv")


if __name__ == "__main__":
    main()