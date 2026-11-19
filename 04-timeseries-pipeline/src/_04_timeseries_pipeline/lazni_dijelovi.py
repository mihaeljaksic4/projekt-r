"""Glumimo dio 5 i dio 6 dok pravi nisu gotovi."""
#isto ai, nije moje

class LazniStateModel:
    """Dio 5. Pamti stanje između poziva, kao pravi State Model."""

    def __init__(self):
        self.trenutno = "IDLE"
        self.prethodno = None
        self.od = None

    def update(self, prozor):
        d = prozor["derived"]
        t = prozor["timestamp"]

        if d["cpu_load_slope"] > 1:
            novo = "LOAD_RISING"
        elif d["cpu_temp_slope"] > 0.2:
            novo = "THERMAL_RESPONSE"
        elif d["cpu_temp_slope"] < -0.2:
            novo = "COOLING"
        elif prozor["context"]["workload"] != "NONE":
            novo = "SUSTAINED_LOAD"
        else:
            novo = "IDLE"

        if self.od is None:
            self.od = t
        if novo != self.trenutno:
            self.prethodno, self.trenutno, self.od = self.trenutno, novo, t

        return {
            "timestamp": t,
            "state": self.trenutno,
            "previous_state": self.prethodno,
            "time_in_state": t - self.od,
            "temperature_trend": d["cpu_temp_slope"],
            "load_trend": d["cpu_load_slope"],
        }


def lazni_prediction_control(poruka):
    """Dio 6. Samo ispiše što je primio."""
    f = poruka["features"]
    print(f'{poruka["timestamp"]}  stanje={poruka["state"]["current_state"]:<16} '
          f'cpu_temp={f["CPU_temp_current"]:.1f}  slope={f["CPU_temp_slope_30s"]:+.2f}  '
          f'workload={f["workload_type"]}  fan={f["fan_profile"]}')