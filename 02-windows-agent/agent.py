import asyncio
import json
import time
import psutil

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

import monitor
import upravljanje

ADRESA = "ws://localhost:8000/ws/agent"

NAREDBE = {
    "start_workload": upravljanje.start_workload,
    "stop_workload": upravljanje.stop_workload,
    "set_affinity": upravljanje.set_affinity,
    "set_priority": upravljanje.set_priority,
    "suspend_process": upravljanje.suspend_process,
    "resume_process": upravljanje.resume_process,
    "get_process_metrics": upravljanje.get_process_metrics,
}

def napravi_poruku(message_type, **polja):
    poruka = {"message_type": message_type, "timestamp": time.time(), "source": "system-agent"}
    poruka.update(polja)
    return json.dumps(poruka)


async def obradi_naredbu(ime, args):
    funkcija = NAREDBE.get(ime)
    if funkcija is None:
        return {"ok": False, "error": f"Nepoznata naredba: {ime}"}
    try:
        rezultat = await asyncio.to_thread(funkcija, **args)
        return {"ok": True, "result": rezultat}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

async def salji_metrike(ws):
    psutil.cpu_percent(percpu=True)
    prev_disk = psutil.disk_io_counters()
    prev_t = time.time()
    
    try:   
        while True:
            await asyncio.sleep(1)
            uzorak, prev_disk, prev_t = monitor.collect(prev_disk, prev_t)
            vrijeme = uzorak.pop("timestamp")
            await ws.send(napravi_poruku("telemetry", timestamp=vrijeme, metrics=uzorak))
    except ConnectionClosed:
        pass

        
async def primaj_naredbe(ws):
    async for tekst in ws:
        poruka = json.loads(tekst)
        if poruka.get("message_type") != "command":
            continue
        ime = poruka["cmd"]
        args = poruka.get("args", {})
        print(f"<-- NAREDBA {poruka['id']} {ime} {args}")
        odgovor = await obradi_naredbu(ime, args)
        print(f"--> ODGOVOR {poruka['id']} {odgovor}")
        await ws.send(napravi_poruku("response", id=poruka["id"], **odgovor))
    
async def main():
    while True:
        try:
            async with connect(ADRESA) as ws:
                print(f"Spojen na {ADRESA}")
                metrike = asyncio.create_task(salji_metrike(ws))
                try:
                    await primaj_naredbe(ws)
                finally:
                    metrike.cancel()
            print("Server je zatvorio vezu.")
        except (OSError, ConnectionClosed) as e:
            print(f"Nema veze sa serverom ({type(e).__name__}).")
        print("Pokušavam ponovno za 3 s...")
        await asyncio.sleep(3)
    
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent zaustavljen.")
    finally:
        for sid in list(upravljanje.workloadi):
            if upravljanje.workload_radi(sid):
                print(f"Gasim workload {sid}: {upravljanje.stop_workload(sid)}")