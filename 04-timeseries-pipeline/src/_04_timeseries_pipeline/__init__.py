import pandas


from .generator import generiraj_uzorke, generiraj_evente
from .obrada import obradi
from .pipeline import napravi_prozor, poruka_za_dio5, feature_vector, poruka_za_dio6, napravi_za_testing
from .lazni_dijelovi import lazni_prediction_control
from .state_model_interface import on_pipeline_packet
from .pregled import spremi_pregled

class UtijekuObrada:

    def __init__(self):
        self.uzorci = []
        self.eventi = []
        self.redci = []
        self.zadnja_obradena = None

    def novi_uzorak(self, u):
        self.uzorci.append(u)

    def novi_event(self, e):
        self.eventi.append(e)

    def korak(self, sada):

        if not self.uzorci:
            print("Nema prikupljenih uzoraka.")
            return None

        #obrada svih uzoraka
        w = obradi(self.uzorci, self.eventi)
        #spremi_pregled("05_podaci_s_postavkama", w)


        #vreme u pandas format
        sada_dt = pandas.to_datetime(sada, unit="s")

        #zememo zadnjih 60 sekundi za obradu
        w = w.loc[w.index > sada_dt - pandas.Timedelta(seconds=60)]

        w = w.loc[w.index < sada_dt.floor("1s")]      
   

        if w.empty:
            return None

        t = w.index[-1]

        if t == self.zadnja_obradena:
            return None

        spremi_pregled("05_podaci_s_postavkama", w.loc[[t]])

        #prozor od 30 sek
        djelic = napravi_prozor(w, t)

        if djelic is None:
            return None


        #slanje podataka State Modelu
        porukaprema5 = poruka_za_dio5(djelic)
        spremi_pregled("state_model_ulaz",
            {
                "timestamp": porukaprema5["timestamp"],
                "columns": str(porukaprema5["columns"]),
                "sequence": str(porukaprema5["sequence"]),
                "derived": str(porukaprema5["derived"]),
                "context": str(porukaprema5["context"])
            }
        )


        natragod5 = on_pipeline_packet(porukaprema5)
        spremi_pregled(
            "state_model_izlaz",
            {
                "timestamp": porukaprema5["timestamp"],
                **natragod5
            }
        )


        #feature vektor
        features = feature_vector(djelic)
        

        #slanje podataka Prediction i Control
        porukaprema6 = poruka_za_dio6(djelic, features, natragod5)
        spremi_pregled(
            "prediction_control_ulaz",
            {
                "timestamp": porukaprema6["timestamp"],
                **features,
                "current_state": natragod5["current_state"],
                "previous_state": natragod5["previous_state"],
                "time_in_state": natragod5["time_in_state"],
                "state_probabilities": str(natragod5["state_probabilities"]),
                "transition_history": str(natragod5["transition_history"])
            }
        )


        rezultat = lazni_prediction_control(porukaprema6)

        #spremanje rezultata
        self.redci.append((t, features, natragod5))

        self.zadnja_obradena = t

        return rezultat

    def spremi_csv(self):

        if not self.redci:
            print("Nema rezultata za spremiti.")
            return

        w = obradi(self.uzorci, self.eventi)

        tablica = napravi_za_testing(w, self.redci)

        tablica.to_csv("rezultati.csv")

        #print("Rezultati spremljeni u rezultati.csv")


def main():
    pocetak = 1730000000.0

    svi_uzorci = generiraj_uzorke(trajanje=120, pocetak=pocetak)
    svi_eventi = generiraj_evente(pocetak)

    spremi_pregled("01_sirovi_uzorci", pandas.DataFrame([{"timestamp": u["timestamp"], **u["metrics"]} for u in svi_uzorci ]))

    spremi_pregled("02_eventi", svi_eventi)




    obrada_trenutak = UtijekuObrada()
    indx_uzorka = 0
    indx_eventa = 0

    #simulacija dobivanja podataka svake sekunde
    for s in range(120):
        sada = pocetak+s+1

        #dosli do ove sekunde
        while ( indx_uzorka < len(svi_uzorci) and svi_uzorci[indx_uzorka]["timestamp"] < sada ):
            obrada_trenutak.novi_uzorak(svi_uzorci[indx_uzorka])
            indx_uzorka+=1

        #dosli eventi
        while ( indx_eventa < len(svi_eventi) and svi_eventi[indx_eventa]["timestamp"] < sada ):
            obrada_trenutak.novi_event(svi_eventi[indx_eventa])
            indx_eventa+=1

        #obrada
        obrada_trenutak.korak(sada)


    obrada_trenutak.spremi_csv()
