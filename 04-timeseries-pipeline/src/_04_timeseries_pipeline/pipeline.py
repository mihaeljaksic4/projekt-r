import numpy
import pandas

STUPCI = ["cpu_usage", "gpu_usage", "ram_usage", "disk_io", "cpu_temp", "gpu_temp", "fan_rpm"]
KONTEKST = ["workload", "fan_profile", "performance_profile", "power_plan"]
PROZOR = 30


def slope(vrijednosti):
    return float(numpy.polyfit(numpy.arange(len(vrijednosti)), vrijednosti, 1)[0])


def napravi_prozor(w, t):
    #za sekundu t zadnjih 30 redaka - None ako prozor ima rupe
    kraj = w.index.get_loc(t)
    if kraj + 1 < PROZOR: #nema puno 30s jos za taj dio
        return None
    djelic = w.iloc[kraj + 1 - PROZOR: kraj + 1]
    if djelic["rupa"].any():
        return None
    return djelic


def poruka_za_dio5(djelic):
    #izlaz A - sekvencijalni prozor za State Model
    return {
        "timestamp": djelic.index[-1].tz_localize("UTC").isoformat(),#djelic.index[-1].isoformat(), #djelic.index[-1].timestamp(),
        "columns": STUPCI,
        "sequence": djelic[STUPCI].values.tolist(),
        "derived": {
            "cpu_load_slope": slope(djelic["cpu_usage"].values),
            "cpu_temp_slope": slope(djelic["cpu_temp"].values),
            "gpu_temp_slope": slope(djelic["gpu_temp"].values),
            "fan_rpm_slope": slope(djelic["fan_rpm"].values),
        },
        "context": {k: str(djelic[k].iloc[-1]) for k in KONTEKST},
    }


def feature_vector(djelic):
    #izlaz B - jedan redak za Prediction

    zadnji = djelic.iloc[-1]
    prosjek = djelic.mean(numeric_only=True)

    f= {
        "CPU_load_current": float(zadnji["cpu_usage"]),
        "CPU_load_mean_30s": float(prosjek["cpu_usage"]),
        "CPU_load_slope_30s": slope(djelic["cpu_usage"].values),

        "GPU_load_current": float(zadnji["gpu_usage"]),
        "GPU_load_mean_30s": float(prosjek["gpu_usage"]),

        "CPU_temp_current": float(zadnji["cpu_temp"]),
        "CPU_temp_mean_30s": float(prosjek["cpu_temp"]),
        "CPU_temp_slope_30s": slope(djelic["cpu_temp"].values),

        "GPU_temp_current": float(zadnji["gpu_temp"]),
        "GPU_temp_slope_30s": slope(djelic["gpu_temp"].values),

        "fan_RPM_current": float(zadnji["fan_rpm"]),
        "fan_RPM_mean_30s": float(prosjek["fan_rpm"]),

        "RAM_current": float(zadnji["ram_usage"]),
        "disk_IO_mean_30s": float(prosjek["disk_io"]),

        "workload_type": str(zadnji["workload"]),
        "fan_profile": str(zadnji["fan_profile"]),
        "performance_profile": str(zadnji["performance_profile"]),
        "power_plan": str(zadnji["power_plan"]),
    }

    return f

def poruka_za_dio6(djelic, features, stanje_izlaz):
    #fv + izlaz z statemodela
    return {
        "timestamp": djelic.index[-1].tz_localize("UTC").isoformat(), #djelic.index[-1].isoformat(), #djelic.index[-1].timestamp(),
        "features": features,
        "state": {
             "current_state": stanje_izlaz["current_state"],
             "previous_state": stanje_izlaz["previous_state"],
             "time_in_state": stanje_izlaz["time_in_state"],
             "state_probabilities": stanje_izlaz["state_probabilities"]
        }
    }

STANJA = ["IDLE", "LOAD_RISING", "SUSTAINED_LOAD", "THERMAL_RESPONSE", "THERMAL_STEADY", "COOLING"]
def napravi_za_testing(w, redci, session_id="sesija"):
    #izlaz C - za 1 sec t feature vector + kaj se dogodilo v t+30
    #redci su lista (timestamp, features, izlaz od 5. djela)
    stanja = {t: izlazod5["current_state"] for t, _, izlazod5 in redci}
    rezultati = []
    for t, features, izlazod5 in redci:

        buduce_vrijeme = t + pandas.Timedelta(seconds=30)

        if buduce_vrijeme not in w.index:
            continue

        buduci_redak = w.loc[buduce_vrijeme]

        if buduci_redak["rupa"]:
            continue

        if buduce_vrijeme not in stanja:                  
            continue
        vjerojatnosti = izlazod5.get("state_probabilities") or {} 

        redak = {
            "timestamp": t,
            **features,
            "current_state": izlazod5["current_state"],
            "previous_state": izlazod5["previous_state"] or "NONE",
            "time_in_state": izlazod5["time_in_state"],
            **{f"p_{x}": float(vjerojatnosti.get(x, 0.0)) for x in STANJA},
            "target_state_t_plus_30": stanja[buduce_vrijeme],
            "target_cpu_temp_t_plus_30": float(buduci_redak["cpu_temp"]),
            "target_gpu_temp_t_plus_30": float(buduci_redak["gpu_temp"]),
            "session_id": session_id
        }

        rezultati.append(redak)

    pomocni = pandas.DataFrame(rezultati)
    if not pomocni.empty:
        pomocni = pomocni.set_index("timestamp")

    return pomocni