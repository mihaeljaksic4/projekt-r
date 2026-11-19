import random
#isto ai, nije moje
def generiraj_uzorke(trajanje=120, pocetak=1730000000.0, seed=1):
    """Lažni dio 3: lista uzoraka u formatu iz dokumenta."""
    random.seed(seed)
    uzorci = []
    temp = 45.0

    for s in range(trajanje):
        # opterećenje CPU-a: mirno, rampa, visoko, pad
        if s < 10:
            cpu = 10
        elif s < 25:
            cpu = 10 + (s - 10) * 80 / 15
        elif s < 100:
            cpu = 90
        else:
            cpu = 15
        cpu += random.uniform(-3, 3)

        cilj = 45 + 0.45 * cpu

        temp += (cilj - temp) * 0.08
        gpu_temp = temp - 5 + random.uniform(-0.5, 0.5)
        fan = 1800 + 45 * (temp - 45) + random.uniform(-30, 30)

        t = pocetak + s

        # Dodan ram (memory_usage) uz cpu_usage
        uzorci.append({"timestamp": t + 0.03, "source": "system-agent",
                       "metrics": {
                           "cpu_usage": round(cpu, 1),
                           "ram_usage": round(random.uniform(40, 65), 1)  # Simulacija RAM-a (npr. 40-65%)
                       }})
        uzorci.append({"timestamp": t + 0.05, "source": "system-agent",
                       "metrics": {"gpu_usage": round(50 + s * 0.05, 1)}})
        uzorci.append({"timestamp": t + 0.07, "source": "system-agent",
                       "metrics": {"disk_io": round(random.uniform(5, 15), 1)}})

        cpu_temp = round(temp + random.uniform(-0.5, 0.5), 1)
        if s == 55:
            cpu_temp = 250.0                          # nemoguća vrijednost
        if not (40 <= s <= 42 or 70 <= s <= 89):      # kratka i duga rupa
            uzorci.append({"timestamp": t + 0.21, "source": "omen-adapter",
                           "metrics": {"cpu_temp": cpu_temp}})
        uzorci.append({"timestamp": t + 0.22, "source": "omen-adapter",
                       "metrics": {"gpu_temp": round(gpu_temp, 1)}})
        uzorci.append({"timestamp": t + 0.24, "source": "omen-adapter",
                       "metrics": {"fan_rpm": round(fan)}})

    return uzorci


def generiraj_evente(pocetak=1730000000.0):
    """Lažni dio 3: lista događaja u formatu iz dokumenta."""
    return [
        {"timestamp": pocetak + 10, "type": "workload_started",
         "workload": "cpu_stress", "parameters": {"duration": 90}},
        {"timestamp": pocetak + 11, "type": "process_started", "process": "python.exe"},
        {"timestamp": pocetak + 20, "type": "resource_limit_applied",
         "parameters": {"max_cpu_percent": 80}},
        {"timestamp": pocetak + 50, "type": "fan_profile_changed", "value": "MAX"},
        {"timestamp": pocetak + 100, "type": "workload_stopped", "workload": "cpu_stress"},
    ]

