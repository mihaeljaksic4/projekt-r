"""Treniranje modela: stanje i temperature 30 s unaprijed."""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PUTANJA_PODATAKA = "data/sessions.csv"
PUTANJA_MODELA = "models/predictor.joblib"
HORIZONT = 30

NUMERICKE = [
    "cpu_usage", "gpu_usage", "cpu_temp", "gpu_temp", "fan_rpm", "ram_usage", "disk_io",
    "cpu_usage_mean", "gpu_usage_mean", "cpu_temp_mean", "fan_rpm_mean", "disk_io_mean",
    "cpu_load_slope", "cpu_temp_slope", "gpu_temp_slope", "fan_rpm_slope",
    "time_in_state",
]
KATEGORICKE = [
    "workload", "fan_profile", "performance_profile", "power_plan",
    "current_state", "previous_state",
]
ZNACAJKE = NUMERICKE + KATEGORICKE


def dodaj_ciljeve(podaci):
    podaci = podaci.sort_values(["session_id", "timestamp"])
    buducnost = podaci.groupby("session_id").shift(-HORIZONT)

    podaci["cilj_stanje"] = buducnost["current_state"]
    podaci["cilj_promjena_cpu"] = buducnost["cpu_temp"] - podaci["cpu_temp"]
    podaci["cilj_promjena_gpu"] = buducnost["gpu_temp"] - podaci["gpu_temp"]
    return podaci.dropna(subset=["cilj_stanje"])


def napravi_model(procjenitelj):
    predobrada = ColumnTransformer([
        ("numericke", StandardScaler(), NUMERICKE),
        ("kategoricke", OneHotEncoder(handle_unknown="ignore"), KATEGORICKE),
    ])
    return make_pipeline(predobrada, procjenitelj)


def treniraj_modele(podaci):
    x = podaci[ZNACAJKE]
    return {
        "cpu": napravi_model(LinearRegression()).fit(x, podaci["cilj_promjena_cpu"]),
        "gpu": napravi_model(LinearRegression()).fit(x, podaci["cilj_promjena_gpu"]),
        "stanje": napravi_model(LogisticRegression(max_iter=3000, class_weight="balanced"))
                  .fit(x, podaci["cilj_stanje"]),
    }


def evaluiraj(modeli, test):
    x = test[ZNACAJKE]
    for dio in ["cpu", "gpu"]:
        stvarna_promjena = test[f"cilj_promjena_{dio}"]
        greska_modela = mean_absolute_error(stvarna_promjena, modeli[dio].predict(x))
        greska_baselinea = mean_absolute_error(stvarna_promjena, [0] * len(test))
        print(f"{dio} temperatura  MAE: model {greska_modela:.2f} °C, baseline {greska_baselinea:.2f} °C")

    tocnost_modela = accuracy_score(test["cilj_stanje"], modeli["stanje"].predict(x))
    tocnost_baselinea = accuracy_score(test["cilj_stanje"], test["current_state"])
    print(f"stanje         točnost: model {tocnost_modela:.2f}, baseline {tocnost_baselinea:.2f}")


def main():
    podaci = dodaj_ciljeve(pd.read_csv(PUTANJA_PODATAKA))

    # cijele sesije idu ili u učenje ili u test
    podjela = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    indeksi_ucenje, indeksi_test = next(podjela.split(podaci, groups=podaci["session_id"]))
    ucenje, test = podaci.iloc[indeksi_ucenje], podaci.iloc[indeksi_test]

    evaluiraj(treniraj_modele(ucenje), test)

    modeli = treniraj_modele(podaci)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"modeli": modeli, "znacajke": ZNACAJKE}, PUTANJA_MODELA)
    print(f"Modeli spremljeni u {PUTANJA_MODELA}")


if __name__ == "__main__":
    main()
