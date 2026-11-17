import asyncio
import json
import time
import psutil

import monitor

async def salji_metrike():              #svake sekunde mjeri i šalji metrike
    psutil.cpu_percent(percpu=True)
    prev_disk = psutil.disk_io_counters()
    prev_t = time.time()
    
    while True:
        await asyncio.sleep(1)
        uzorak, prev_disk, prev_t = monitor.collect(prev_disk, prev_t)
        print("METRIKE", json.dumps(uzorak))
    
async def primaj_naredbe():             #glumimo naredbe; svakih 5s "stigne" jedna
    broj = 0
    
    while True:
        await asyncio.sleep(5)
        broj += 1
        print(f"NAREDBA #{broj} stigla")
        
async def main():
    await asyncio.gather(salji_metrike(), primaj_naredbe())
    
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent zaustavljen.")