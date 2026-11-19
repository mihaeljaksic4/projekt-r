# 6. Prediction & Control

Predviđa stanje sustava i CPU/GPU temperature za 30 s unaprijed te na temelju toga bira upravljačku akciju.

## Datoteke

| Datoteka | Što radi |
|---|---|
| `src/train.py` | iz snimljenih sesija napravi targete za t+30, istrenira Linear i Logistic Regression, spremi `models/predictor.joblib` |
| `src/predictor.py` | feature vector → predviđeno stanje i temperature za 30 s |
| `src/controller.py` | trenutno stanje + predikcija → jedna akcija ili `NO_ACTION` |
| `src/executors.py` | šalje akciju kao HTTP zahtjev modulu 01 (HW adapter) ili 02 (Windows agent) |
| `src/service.py` | petlja koja sve povezuje |

```
vektor značajki → Prediktor → Upravljac → izvrsi → modul 01 / 02
```

## Ulaz: feature vector (jednom u sekundi)

Jedan ravni dict, isti stupci za treniranje (CSV) i za rad. Nazivi prate Pipeline (04).

| Grupa | Stupci |
|---|---|
| identifikacija | `timestamp`, `session_id` (samo u CSV-u) |
| CPU / GPU opterećenje | `CPU_load_current`, `CPU_load_mean_30s`, `CPU_load_slope_30s`, `GPU_load_current`, `GPU_load_mean_30s` |
| temperature | `CPU_temp_current`, `CPU_temp_mean_30s`, `CPU_temp_slope_30s`, `GPU_temp_current`, `GPU_temp_slope_30s` |
| ventilator, RAM, disk | `fan_RPM_current`, `fan_RPM_mean_30s`, `RAM_current`, `disk_IO_mean_30s` |
| kontekst | `workload_type`, `fan_profile`, `performance_profile`, `power_plan`, opcionalno `workload_pid` |
| State Model | `current_state`, `previous_state`, `time_in_state` |

## Izlaz

Predikcija:
```json
{
  "predicted_state_30s": "THERMAL_RESPONSE",
  "state_probabilities": {"THERMAL_RESPONSE": 0.72, "SUSTAINED_LOAD": 0.18, "IDLE": 0.1},
  "predicted_cpu_temperature_30s": 86.4,
  "predicted_gpu_temperature_30s": 79.8
}
```

Akcija:
```json
{"action": "set_fan_profile", "target": "hw_adapter", "params": {"profile": "MAX"}}
```

## Slanje akcija modulima 01 i 02

Modul 06 ne izvršava akcije sam, nego ih šalje kao HTTP zahtjev modulu koji je za njih
zadužen. Taj modul izvrši promjenu i javi platformi 03 događaj (npr. `fan_profile_changed`),
tako da Pipeline zna da se postavka promijenila.

```
POST http://<adresa modula>/<action>      tijelo: params kao JSON
```

| Modul | Adresa (prijedlog) | Primjer |
|---|---|---|
| 01 HW adapter | `http://127.0.0.1:8001` | `POST /set_fan_profile {"profile": "MAX"}` |
| 02 Windows agent | `http://127.0.0.1:8002` | `POST /set_priority {"pid": 4812, "razina": "below_normal"}` |

Očekivani odgovor: `{"status": "ok"}`, ili HTTP greška ako akcija nije uspjela.
Ako modul ne radi, akcija se preskoči uz poruku `[NEDOSTUPNO]`; upravljač nastavlja raditi.

## Pravila controllera

Provjeravaju se redom, prvo ispunjeno odlučuje; nakon akcije 30 s nema novih akcija.

1. predviđena T ≥ 95 °C → ventilator na `MAX`
2. sada `LOAD_RISING`/`SUSTAINED_LOAD`, za 30 s `THERMAL_RESPONSE`, T_cpu > 85 °C → ventilator jače
3. ventilator već na `MAX`, predviđena T > 85 °C → niži prioritet workloada
4. sada i za 30 s `IDLE`/`COOLING`, T < 60 °C → ventilator tiše

## Pokretanje

```bash
pip install -r requirements.txt
python src/train.py
python src/service.py
```

Sve naredbe se pokreću iz mape `06-prediction-control` (putanje su relativne).

Putanje i postavke (`PUTANJA_PODATAKA`, `PROBNI_RAD`, …) su konstante na vrhu datoteka.
`PROBNI_RAD = True` znači da se akcije samo ispisuju.

## TODO

- moduli 01 i 02 trebaju HTTP endpoint za akcije (adrese i portovi su prijedlog)
- nazivi fan profila prema modulu 01
- podaci uživo umjesto čitanja CSV-a
