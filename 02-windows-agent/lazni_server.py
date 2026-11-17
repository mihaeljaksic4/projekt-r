import asyncio
import json
import time

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

HOST = "localhost"
PORT = 8000

brojac = 0

async def posalji_naredbu(ws, cekanja, cmd, **args):
    global brojac
    brojac += 1
    id_naredbe = f"c-{brojac}"
    
    odgovor = asyncio.get_running_loop().create_future()
    cekanja[id_naredbe] = odgovor
    
    poruka = {"message_type": "command", "timestamp": time.time(), "source": "control", "id": id_naredbe, "cmd": cmd, "args": args}
    print(f">>> NAREDBA {id_naredbe} {cmd} {args}")
    await ws.send(json.dumps(poruka))
    
    return await odgovor

async def scenarij(ws, cekanja):
    await asyncio.sleep(2)
    odg = await posalji_naredbu(ws, cekanja, "start_workload", br_jezgri=2, trajanje=30)
    pid = odg["result"]["pid"]
    sid = odg["result"]["session_id"]
    
    await asyncio.sleep(4)
    await posalji_naredbu(ws, cekanja, "set_affinity", pid=pid, jezgre=[0])
    
    await asyncio.sleep(4)
    await posalji_naredbu(ws, cekanja, "suspend_process", pid = pid)
    
    await asyncio.sleep(3)
    await posalji_naredbu(ws, cekanja, "nepostojeca")
    
    await asyncio.sleep(1)
    await posalji_naredbu(ws, cekanja, "stop_workload", session_id=sid)
    
    print("=== scenarij gotov, i dalje primam telemetriju (CTRL + C za kraj) ===")
    
async def agent_spojen(ws):
    print("=== agent spojen")
    cekanja = {}
    
    async def citaj():
        try:
            async for tekst in ws:
                poruka = json.loads(tekst)
                if poruka["message_type"] == "telemetry":
                    m = poruka["metrics"]
                    print(f"    telemetrija: cpu={m['cpu_usage']}% ram={m['ram_usage']}%")
                elif poruka["message_type"] == "response":
                    print(f"<<< ODGOVOR {poruka['id']} ok={poruka['ok']} {poruka.get('result') or poruka.get('error')}")
                    odgovor = cekanja.pop(poruka["id"], None)
                    if odgovor:
                        odgovor.set_result(poruka)
        except ConnectionClosed:
            pass
        
    citac = asyncio.create_task(citaj())
    try:
        await scenarij(ws, cekanja)
        await citac
    except ConnectionClosed:
        pass
    print("=== agent odspojen")
    
async def main():
    async with serve(agent_spojen, HOST, PORT):
        print(f"Lažni server sluša na ws://{HOST}:{PORT}")
        await asyncio.Future()
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer zaustavljen")