import asyncio
import json
import time
import websockets

#testni klijent koji simulira slanje telemetrije iz 2. dijela
#spaja se na moj WebSocket endpoint i šalje jedno izmišljeno mjerenje

async def posalji_mjerenje():
    poruka = {
        "message_type": "telemetry",
        "timestamp": time.time(),
        "source": "test-agent",
        "metrics": {
            "cpu_usage": 10,
            "gpu_usage": 35,
            "ram_usage": 61.3,
            "disk_io": 12.4,
            "cpu_temp": 75,
            "gpu_temp": 60,
            "fan_rpm": 2500
        }
    }

    async with websockets.connect(
        "ws://127.0.0.1:8000/ws/agent"
    ) as websocket:
        await websocket.send(json.dumps(poruka))
        print("Mjerenje poslano.")


asyncio.run(posalji_mjerenje())