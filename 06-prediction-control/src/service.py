"""Glavna petlja: predikcija -> izbor akcije -> slanje akcije. Zasad čita podatke iz CSV-a."""

import time

import pandas as pd

from controller import Upravljac
from executors import izvrsi
from predictor import Prediktor

PUTANJA_PODATAKA = "data/sessions.csv"
PROBNI_RAD = True
SEKUNDI_PO_RETKU = 1


def main():
    prediktor = Prediktor()
    upravljac = Upravljac()
    retci = pd.read_csv(PUTANJA_PODATAKA).sort_values(["session_id", "timestamp"])

    for _, redak in retci.iterrows():
        vektor = redak.to_dict()
        sada = pd.Timestamp(vektor["timestamp"]).timestamp()

        predikcija = prediktor.predvidi(vektor)
        akcija = upravljac.odluci(vektor, predikcija, sada)
        rezultat = izvrsi(akcija, probni_rad=PROBNI_RAD)

        print(vektor["timestamp"], predikcija["predicted_state_30s"],
              predikcija["predicted_cpu_temperature_30s"], akcija["action"], rezultat)

        time.sleep(SEKUNDI_PO_RETKU)


if __name__ == "__main__":
    main()
