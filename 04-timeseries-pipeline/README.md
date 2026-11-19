# Time-Series Pipeline


Time-Series Pipeline uzima sirovu telemetriju i događaje, sinkronizira i čisti, dodaje kontekst i događaje, izrađuje vremenske prozore te priprema dva izlaza:

1.  sekvencijalni ulaz za State Model
2.  feature vector zajedno s kontekstom State Modela za Prediction & Control

Priprema podatke za treniranje i testiranje Prediction modela.  


Pipeline dobiva listu uzoraka čiji je oblik:  
```
{
    "timestamp": 1730000000.03,
    "metrics": {
        "cpu_usage": 10.0,
        "gpu_usage": 10.0,
        "ram_usage": 10.0,
        "disk_io": 100.0,
        "cpu_temp": 20.0,
        "gpu_temp": 20.0,
        "fan_rpm": 900.0
    }
}

{
    "timestamp": "...", 
    "type": "workload_started", 
    "workload": "cpu_stress", 
}
```
Jedinice: °C, %, rpm, s; disk u MB/s; trend (slope) u jedinici/s.  

kratko popunjavanje rupa u podacima - max 3s, prozor - 30s   


Obrada: vrijednosti izvan dopuštenih raspona postavljaju se na NaN, kratke rupe popunjavaju se forward fill, veće rupe označavam u stupcu s nazivom rupa, stupac event ako nema događaja piše NONE.  
Ulaz: lista uzoraka i lista događaja  
Izlaz: obrađeni DataFrame s vremenskim indeksom, vrijednostima, kontekstom, događajima i oznakom rupa  


### Vremenski prozor

Ulaz: obrađena vremenska tablica w i završni trenutak prozora t   
Uzima se posljednjih 30 redaka, prozor se ne vraća ako nema dovoljno redaka ili ako sadrži redak koji ima rupu.   
Izlaz: DataFrame s 30 uzastopnih redaka za daljnju obradu ili None

Za State Model ulaz:
```
{
    "timestamp": "...",
    "columns": [
        "cpu_usage",
        "gpu_usage",
        "ram_usage",
        "disk_io",
        "cpu_temp",
        "gpu_temp",
        "fan_rpm"
    ],
    "sequence": [
        ukupno 30 redaka
    ],
    "derived": {
        "cpu_load_slope": 0.0,
        "cpu_temp_slope": 0.0,
        "gpu_temp_slope": 0.0,
        "fan_rpm_slope": 0.0
    },
    "context": {
        "workload": "...",
        "fan_profile": "...",
        "performance_profile": "...",
        "power_plan": "..."
    }
}
```

U projektnom prijedlogu State Modelov prozor je opisan s window_start i window_end, ali moja funkcija šalje samo završni timestamp.

State Model vraća:
``` 
{
    "current_state": result["current_state"],
    "previous_state": result["previous_state"],
    "time_in_state": result["time_in_state"],
    "state_probabilities": result["state_probabilities"],
    "transition_history": result["transition_history"]
}
```



### Feature vector


Ulaz: ispravan prozor od 30 sekundi  
Izlaz: rječnik

Fv:   
  CPU_load_current      
  CPU_load_mean_30s    
  CPU_load_slope_30s   
  GPU_load_current     
  GPU_load_mean_30s    
  CPU_temp_current    
  CPU_temp_mean_30s   
  CPU_temp_slope_30s    
  GPU_temp_current     
  GPU_temp_slope_30s   
  fan_RPM_current      
  fan_RPM_mean_30s     
  RAM_current           
  disk_IO_mean_30s     
  workload_type         
  fan_profile           
  performance_profile  
  power_plan          


Ulaz za Prediction:
```
{
    "timestamp": "...",
    "features": {
        feature vector
    },
    "state": {
        "current_state": "...",
        "previous_state": "...",
        "time_in_state": 0,
        "state_probabilities": {
            vjerojatnosti stanja
        }
    }
}
```

Nedostaje transition_history.

Prema prijedlogu za svaki trenutak t Pipeline treba povezati  feature vector i State Model kontekst u trenutku t s ciljnim vrijednostima u budućnosti tj. stanje u t + 30, CPU temperaturu u t + 30 i GPU temperaturu u t + 30.
```
{
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
```