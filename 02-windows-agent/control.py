import sys
import psutil
import time

MB = 1024 * 1024

workload = psutil.Popen([sys.executable, "cpu_workload.py", "--brJezgri", "2", "--trajanje", "20"])
print("Pokrenut workload, PID:", workload.pid)

procesi = {}
t = 0

while workload.poll() is None:
    time.sleep(1)
    if workload.poll() is not None:
        break
    t += 1
    
    svi = [workload] + workload.children(recursive=True)
    
    if t == 4:
        for p in svi:
            try:
                p.cpu_affinity([0])
            except psutil.NoSuchProcess:
                pass
        print("--> affinity: samo jezgra 0")
        
    if t == 6:
        for p in svi:
            try:
                if psutil.WINDOWS:
                    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    p.nice(10)
            except psutil.NoSuchProcess:
                pass
        print("--> prioritet snizen")
    
    cpu = 0.0
    memorija = 0
    for p in svi:
        p = procesi.setdefault(p.pid, p)
        try:
            cpu += p.cpu_percent()
            memorija += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    
    print(f"t={t}s -- Procesa: {len(svi)} -- CPU workloada: {cpu:.1f}% -- Prioritet: {workload.nice()} -- RAM: {memorija / MB:.1f} MB")
    
print("Workload je zavrsio.")