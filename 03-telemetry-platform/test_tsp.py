import asyncio

from websockets.asyncio.client import connect


async def main():
    async with connect("ws://127.0.0.1:8000/ws/stream") as ws:
        await ws.send("get_latest")
        odgovor = await ws.recv()
        print(odgovor)


if __name__ == "__main__":
    asyncio.run(main())