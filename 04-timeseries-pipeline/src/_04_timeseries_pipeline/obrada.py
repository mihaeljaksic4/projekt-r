#sirovi uzorci + eventi  ->  uredna tablica - 1 redak = 1 sek

import pandas

SANITY = {
    "cpu_usage": (0, 100),
    "gpu_usage": (0, 100),
    "ram_usage": (0, 100),
    "disk_io": (0, 10000),
    "cpu_temp": (0, 110),
    "gpu_temp": (0, 110),
    "fan_rpm": (0, 7000)
}

POCETNE_POSTAVKE = {
    "workload": "NONE",
    "fan_profile": "BALANCED",
    "performance_profile": "BALANCED",
    "power_plan": "BALANCED",
}

MAX_RUPA_SEK = 3

def uzorci_u_tablicu(uzorci):
    redci = []
    for uzorak in uzorci:
        redak = {"timestamp": uzorak["timestamp"]}
        redak.update(uzorak["metrics"])
        redci.append(redak)

    tablica = pandas.DataFrame(redci)

    #broj v pravo vrijeme
    tablica["timestamp"] = pandas.to_datetime(tablica["timestamp"], unit="s")

    tablica = tablica.set_index("timestamp").sort_index()
    tablica = tablica.resample("1s").mean()

    return tablica


def izbaci_nemoguce_vrijednosti(tablica):
    for kanal, (najmanje, najvise) in SANITY.items():
        if kanal in tablica.columns:
            rasponu = tablica[kanal].between(najmanje, najvise)
            tablica[kanal] = tablica[kanal].where(rasponu)
    return tablica



def popravi_rupe(tablica):
    tablica = tablica.ffill(limit=MAX_RUPA_SEK)
    tablica["rupa"] = tablica.isna().any(axis=1)
    return tablica



def sto_event_mijenja(event):
   #iz eventa kaze koju postavku meja
   #(ime_postavke, nova_vrijednost), None ako ne meja

    vrsta = event["type"]

    if vrsta == "workload_started":
        return "workload", event["workload"].upper()
    if vrsta == "workload_stopped":
        return "workload", "NONE"
    if vrsta == "fan_profile_changed":
        return "fan_profile", event["value"]
    if vrsta == "performance_profile_changed":
        return "performance_profile", event["value"]
    if vrsta == "power_plan_changed":
        return "power_plan", event["value"]

    return None


def dodaj_postavke(tablica, eventi):
    #stupci workload, fan_profile, performance_profile, power_plan
    #postavka - od eventa koji ju je promenil do sljedece promjene

    #popis promjena
    promjene = []
    for event in eventi:
        rezultat = sto_event_mijenja(event)
        if rezultat is not None:
            naziv, vrijednost = rezultat
            promjene.append({"timestamp": event["timestamp"], "naziv": naziv, "vrijednost": vrijednost})

    promjene = pandas.DataFrame(promjene, columns=["timestamp", "naziv", "vrijednost"])
    promjene["timestamp"] = pandas.to_datetime(promjene["timestamp"], unit="s")
    promjene["timestamp"] = promjene["timestamp"].dt.floor("1s")    #zaokruzeno na sekundu

    #po svim sek
    for naziv, pocetna_vrijednost in POCETNE_POSTAVKE.items():
        prom_postavke = promjene[promjene["naziv"] == naziv]
        prom_postavke = prom_postavke.set_index("timestamp")["vrijednost"]
        prom_postavke = prom_postavke[~prom_postavke.index.duplicated(keep="last")]
        #ako ih je vise v istoj sekundi zadnja se deva

        #zadnja promjena prije nje
        tablica[naziv] = prom_postavke.reindex(tablica.index, method="ffill")

        #prije prve promjene vrijedi poc vrijednost
        tablica[naziv] = tablica[naziv].fillna(pocetna_vrijednost)

    return tablica


def dodaj_stupac_event(tablica, eventi):
    #v sekundi v koji se nest dogodilo - naziv eventa, ostali - NONE

    if len(eventi) == 0:
        tablica["event"] = "NONE"
        return tablica

    tablica_eventa = pandas.DataFrame(eventi)
    tablica_eventa["timestamp"] = pandas.to_datetime(tablica_eventa["timestamp"], unit="s").dt.floor("1s")

    #ak v istoj sek desi vise eventa spoje se zarezom
    po_sekundi = tablica_eventa.groupby("timestamp")["type"].agg(", ".join)

    tablica["event"] = po_sekundi.reindex(tablica.index).fillna("NONE")
    return tablica




def obradi(uzorci, eventi):
    #jedan redak po sek, stupci: cpu_usage ..., gap, postavke (workload, ...), event
    if not uzorci:
        return pandas.DataFrame()

    tablica = uzorci_u_tablicu(uzorci)
    tablica = izbaci_nemoguce_vrijednosti(tablica)
    tablica = popravi_rupe(tablica)

    tablica = dodaj_postavke(tablica, eventi)
    tablica = dodaj_stupac_event(tablica, eventi)    #kaj se dogodilo v toj sekundi

    return tablica