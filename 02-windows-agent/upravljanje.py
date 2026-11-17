import os
import sys
import time
import uuid
import psutil

MB = 1024 * 1024
OVAJ_FOLDER = os.path.dirname(os.path.abspath(__file__))

if psutil.WINDOWS:
    PRIORITETI = {
        "idle": psutil.IDLE_PRIORITY_CLASS,
        "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
        "normal": psutil.NORMAL_PRIORITY_CLASS,
        "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
        "high": psutil.HIGH_PRIORITY_CLASS,
    }
else:
    PRIORITETI = {
        "idle": 19,
        "below_normal": 10,
        "normal": 0,
        "above_normal": -5,     #sudo na Linuxu
        "high": -10,            #sudo na Linuxu
    }
    
workloadi = {}
_procesi = {}

def _stablo(pid):               #proces + sva njegova djeca
    root = _procesi.get(pid) or psutil.Process(pid)
    try:
        svi = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return []

    return [_procesi.setdefault(p.pid, p) for p in svi]

def _za_svaki(pid, akcija):      #primijeni akciju na cijelo stablo procesa
    broj = 0
    for p in _stablo(pid):
        try:
            akcija(p)
            broj += 1
        except psutil.NoSuchProcess:
            pass
    
    return broj


def start_workload(br_jezgri, trajanje):  #pokrece CPU workload, vraca session_id i PID
    session_id = "s-" + uuid.uuid4().hex[:8]
    
    skripta = os.path.join(OVAJ_FOLDER, "cpu_workload.py")
    
    workload = psutil.Popen([sys.executable, skripta, "--brJezgri", str(br_jezgri), "--trajanje", str(trajanje)])
    workloadi[session_id] = workload
    
    _procesi[workload.pid] = workload
    
    return {"session_id": session_id, "pid": workload.pid}


def workload_radi(session_id):  #True dok workload radi
    return workloadi[session_id].poll() is None
    
    
def stop_workload(session_id):  #ugasi cijeli workload
    svi = _stablo(workloadi[session_id].pid)
    for p in svi:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    ugaseni, zivi = psutil.wait_procs(svi, timeout=3)
    for p in zivi:
        p.kill()
    
    return {"uredno": len(ugaseni), "silom": len(zivi)}

def set_affinity(pid, jezgre):
    return {"procesa": _za_svaki(pid, lambda p: p.cpu_affinity(jezgre))}

def set_priority(pid, razina):
    vrijednost = PRIORITETI[razina]
    return {"procesa": _za_svaki(pid, lambda p: p.nice(vrijednost))}

def suspend_process(pid):
    return {"procesa": _za_svaki(pid, lambda p: p.suspend())}

def resume_process(pid):
    return {"procesa": _za_svaki(pid, lambda p: p.resume())}

def get_process_metrics(pid):  #CPU, RAM i broj procesa za proces i svu njegovu dijecu
    svi = _stablo(pid)
    cpu = 0.0
    memorija = 0.0
    
    for p in svi:
        try:
            cpu += p.cpu_percent()
            memorija += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    
    return {
        "ts": int(time.time() * 1000),
        "pid": pid,
        "procesa": len(svi),
        "cpu_percent": round(cpu, 1),
        "memory_mb": round(memorija / MB, 1),
    }
    
if __name__ == "__main__":
    w = start_workload(br_jezgri=2, trajanje=20)
    sid, pid = w["session_id"], w["pid"]
    print("Pokrenut workload:", w)
    
    akcije = {
        4: lambda: set_affinity(pid, [0]),
        6: lambda: set_priority(pid, "below_normal"),
        9: lambda: suspend_process(pid),
        12: lambda: resume_process(pid),
        15: lambda: stop_workload(sid),
    }
    
    t = 0
    while workload_radi(sid):
        time.sleep(1)
        t += 1
        if t in akcije:
            print(f"--> t={t}s:", akcije[t]())
        if not workload_radi(sid):
            break
        print(f"t={t}s", get_process_metrics(pid))
        
    print("Workload je zavrsio.")