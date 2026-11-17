#pokretanje i praćenje jednog opterećenja(workloada)


import sys
import psutil
import time

MB = 1024 * 1024

workload = psutil.Popen([sys.executable, "cpu_workload.py", "--brJezgri", "2", "--trajanje", "10"])
print("Pokrenut workload, PID:", workload.pid)

procesi = {}

while workload.poll() is None:
    time.sleep(1)
    if workload.poll() is not None:
        break
    
    svi = [workload] + workload.children(recursive=True)
    cpu = 0.0
    memorija = 0
    for p in svi:
        p = procesi.setdefault(p.pid, p)
        try:
            cpu += p.cpu_percent()
            memorija += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    
    print(f"Procesa: {len(svi)} -- CPU workloada: {cpu:.1f}% -- RAM: {memorija / MB:.1f} MB")
    
print("Workload je zavrsio.")