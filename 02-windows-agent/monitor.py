import psutil
import json
import time

MB = 1024 * 1024

def collect(prev_disk, prev_t):
    po_jezgri = psutil.cpu_percent(percpu=True)
    memorija = psutil.virtual_memory()
    
    trenutno = time.time()
    dt = trenutno - prev_t
    disk = psutil.disk_io_counters()
    
    #ts - trenutak mjerenja (u milisekundama od 1.1.1970.)
    #cpu_usage - kolko je cijeli procesor zauzet [%]
    #cpu_per_core - kolko je svaka logicka jezgra zauzeta posebno [%]
    #memory_usage - kolko je RAM-a zauzeto [%]
    #disk_read_mb_s - kolko se MB u sekundi čita s diska
    #disk_write_mb_s - kolko se MB u sekundi piše na disk
        
    uzorak = {
        "ts": int(trenutno * 1000),
        "cpu_usage": round(sum(po_jezgri) / len(po_jezgri), 1),
        "cpu_per_core": po_jezgri,
        "memory_usage": memorija.percent,
        "disk_read_mb_s": round((disk.read_bytes - prev_disk.read_bytes) / MB / dt, 2),
        "disk_write_mb_s": round((disk.write_bytes - prev_disk.write_bytes) / MB / dt, 2),
    }
    
    return uzorak, disk, trenutno
    
    
def main():
    psutil.cpu_percent(percpu=True)
    prev_disk = psutil.disk_io_counters()
    prev_t = time.time()
    
    
    while True:
        time.sleep(1)
        uzorak, prev_disk, prev_t = collect(prev_disk, prev_t)
        print(json.dumps(uzorak))
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nZaustavljeno.")