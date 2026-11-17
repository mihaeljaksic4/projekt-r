import asyncio
import time
import psutil

import monitor
import upravljanje

NAREDBE = {
    "start_workload": upravljanje.start_workload,
    "stop_workload": upravljanje.stop_workload,
    "set_affinity": upravljanje.set_affinity,
    "set_priority": upravljanje.set_priority,
    "suspend_process": upravljanje.suspend_process,
    "resume_process": upravljanje.resume_process,
    "get_process_metrics": upravljanje.get_process_metrics,
}

async def obradi_naredbu(naredba):
    ime = naredba.get("cmd")
    args = naredba.get("args",{})
    print(f"<-- NAREDBA {ime} {args}")
    
    funkcija = NAREDBE.get(ime)
    if funkcija is None:
        odgovor = {"id": naredba.get("id"), "ok": False, "error": f"nepoznata naredba: {ime}"}
    else:
        try:
            rezultat = await asyncio.to_thread(funkcija, **args)
            odgovor = {"id": naredba.get("id"), "ok": True, "result": rezultat}
        except Exception as e:
            odgovor = {"id": naredba.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}"}
            
    print(f"--> ODGOVOR {odgovor}")
    return odgovor

async def salji_metrike():
    psutil.cpu_percent(percpu=True)
    prev_disk = psutil.disk_io_counters()
    prev_t = time.time()
    
    while True:
        await asyncio.sleep(1)
        uzorak, prev_disk, prev_t = monitor.collect(prev_disk, prev_t)
        print(f"METRIKE cpu={uzorak['cpu_usage']}% jezgre={uzorak['cpu_per_core']}")
        
async def lazni_control():      #glumi dio 6
    await asyncio.sleep(2)
    odg = await obradi_naredbu({"id": "c-1", "cmd": "start_workload", "args": {"br_jezgri": 2, "trajanje": 30}})
    
    pid = odg["result"]["pid"]
    sid = odg["result"]["session_id"]
    
    await asyncio.sleep(4)
    await obradi_naredbu({"id": "c-2", "cmd": "set_affinity", "args": {"pid": pid, "jezgre": [0]}})
    
    await asyncio.sleep(4)
    await obradi_naredbu({"id": "c-3", "cmd": "suspend_process", "args": {"pid": pid}})
    
    await asyncio.sleep(3)
    await obradi_naredbu({"id": "c-4", "cmd": "nepostojeca", "args": {}})
    
    await asyncio.sleep(1)
    await obradi_naredbu({"id": "c-5", "cmd": "stop_workload", "args": {"session_id": sid}})
    
async def main():
    await asyncio.gather(salji_metrike(), lazni_control())
    
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent zaustavljen.")