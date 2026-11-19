"""Predikcija stanja i temperatura 30 s unaprijed."""

import joblib
import pandas as pd


class Prediktor:
    def __init__(self, putanja_modela="models/predictor.joblib"):
        spremljeno = joblib.load(putanja_modela)
        self.modeli = spremljeno["modeli"]
        self.znacajke = spremljeno["znacajke"]

    def predvidi(self, vektor):
        x = pd.DataFrame([vektor])[self.znacajke]

        promjena_cpu = self.modeli["cpu"].predict(x)[0]
        promjena_gpu = self.modeli["gpu"].predict(x)[0]

        model_stanja = self.modeli["stanje"]
        vjerojatnosti = model_stanja.predict_proba(x)[0]
        vjerojatnosti_stanja = {
            stanje: round(float(p), 3) for stanje, p in zip(model_stanja.classes_, vjerojatnosti)
        }

        return {
            "predicted_state_30s": max(vjerojatnosti_stanja, key=vjerojatnosti_stanja.get),
            "state_probabilities": vjerojatnosti_stanja,
            "predicted_cpu_temperature_30s": round(float(vektor["CPU_temp_current"] + promjena_cpu), 1),
            "predicted_gpu_temperature_30s": round(float(vektor["GPU_temp_current"] + promjena_gpu), 1),
        }


if __name__ == "__main__":
    primjer = {
        "CPU_load_current": 85.0, "CPU_load_mean_30s": 78.2, "CPU_load_slope_30s": 1.9,
        "GPU_load_current": 12.8, "GPU_load_mean_30s": 11.5,
        "CPU_temp_current": 77.0, "CPU_temp_mean_30s": 74.1, "CPU_temp_slope_30s": 0.31,
        "GPU_temp_current": 72.0, "GPU_temp_slope_30s": 0.20,
        "fan_RPM_current": 2350.0, "fan_RPM_mean_30s": 2200.0,
        "RAM_current": 55.0, "disk_IO_mean_30s": 10.1,
        "workload_type": "CPU_STRESS", "fan_profile": "MAX",
        "performance_profile": "BALANCED", "power_plan": "BALANCED",
        "current_state": "SUSTAINED_LOAD", "previous_state": "LOAD_RISING", "time_in_state": 23.8,
    }
    print(Prediktor().predvidi(primjer))
