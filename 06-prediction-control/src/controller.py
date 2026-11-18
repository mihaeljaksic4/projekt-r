PROFILI_VENTILATORA = ["QUIET", "BALANCED", "MAX"]  # TODO uskladiti s modulom 01
CEKANJE_SEKUNDI = 30
MIN_SIGURNOST = 0.6


def akcija(naziv, cilj=None, **parametri):
    return {"action": naziv, "target": cilj, "params": parametri}


BEZ_AKCIJE = akcija("NO_ACTION")


def jaci_ventilator(profil):
    i = PROFILI_VENTILATORA.index(profil)
    return PROFILI_VENTILATORA[min(i + 1, len(PROFILI_VENTILATORA) - 1)]


def slabiji_ventilator(profil):
    i = PROFILI_VENTILATORA.index(profil)
    return PROFILI_VENTILATORA[max(i - 1, 0)]


def izaberi_akciju(vektor, predikcija):
    stanje_sada = vektor["current_state"]
    ventilator = vektor["fan_profile"]

    stanje_kasnije = predikcija["predicted_state_30s"]
    if predikcija["state_probabilities"][stanje_kasnije] < MIN_SIGURNOST:
        stanje_kasnije = None

    cpu_kasnije = predikcija["predicted_cpu_temperature_30s"]
    najtoplije_kasnije = max(cpu_kasnije, predikcija["predicted_gpu_temperature_30s"])

    # 1. kritična temperatura
    if najtoplije_kasnije >= 95 and ventilator != "MAX":
        return akcija("set_fan_profile", "hw_adapter", profile="MAX")

    # 2. opterećenje sada, termalni odziv za 30 s
    if (stanje_sada in ("LOAD_RISING", "SUSTAINED_LOAD")
            and stanje_kasnije == "THERMAL_RESPONSE"
            and cpu_kasnije > 85
            and ventilator != "MAX"):
        return akcija("set_fan_profile", "hw_adapter", profile=jaci_ventilator(ventilator))

    # 3. ventilator na maksimumu, a i dalje vruće (TODO: vektor nema workload_pid)
    if ventilator == "MAX" and najtoplije_kasnije > 85 and "workload_pid" in vektor:
        return akcija("set_priority", "windows_agent", pid=vektor["workload_pid"], razina="below_normal")

    # 4. sustav miruje i ostat će miran
    if (stanje_sada in ("IDLE", "COOLING")
            and stanje_kasnije in ("IDLE", "COOLING")
            and najtoplije_kasnije < 60
            and ventilator != PROFILI_VENTILATORA[0]):
        return akcija("set_fan_profile", "hw_adapter", profile=slabiji_ventilator(ventilator))

    return BEZ_AKCIJE


class Upravljac:
    def __init__(self):
        self.vrijeme_zadnje_akcije = None

    def odluci(self, vektor, predikcija, sada):
        izabrana = izaberi_akciju(vektor, predikcija)
        if izabrana == BEZ_AKCIJE:
            return BEZ_AKCIJE

        ceka_se = (self.vrijeme_zadnje_akcije is not None
                   and sada - self.vrijeme_zadnje_akcije < CEKANJE_SEKUNDI)
        if ceka_se:
            return BEZ_AKCIJE

        self.vrijeme_zadnje_akcije = sada
        return izabrana
