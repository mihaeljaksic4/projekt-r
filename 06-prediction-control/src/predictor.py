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
            "predicted_cpu_temperature_30s": round(float(vektor["cpu_temp"] + promjena_cpu), 1),
            "predicted_gpu_temperature_30s": round(float(vektor["gpu_temp"] + promjena_gpu), 1),
        }


if __name__ == "__main__":
    primjer = {
        "cpu_usage": 85.0, "gpu_usage": 12.8, "cpu_temp": 77.0, "gpu_temp": 72.0,
        "fan_rpm": 2350.0, "ram_usage": 55.0, "disk_io": 9.8,
        "cpu_usage_mean": 78.2, "gpu_usage_mean": 11.5, "cpu_temp_mean": 74.1,
        "fan_rpm_mean": 2200.0, "disk_io_mean": 10.1,
        "cpu_load_slope": 1.9, "cpu_temp_slope": 0.31, "gpu_temp_slope": 0.20, "fan_rpm_slope": 12.0,
        "workload": "CPU_STRESS", "fan_profile": "MAX",
        "performance_profile": "BALANCED", "power_plan": "BALANCED",
        "current_state": "SUSTAINED_LOAD", "previous_state": "LOAD_RISING", "time_in_state": 23.8,
    }
    print(Prediktor().predvidi(primjer))
