from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, ValidationError
import uvicorn
import json
from typing import Literal
import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path


#DEFINICIJA MJERENJA:

#klasa Metrics definira fieldove za izmjerene vrijednosti po dogovoru
class Metrics(BaseModel):
    cpu_usage: float 
    ram_usage: float 
    cpu_per_core: list[float] 
    disk_read_mb_s: float 
    disk_write_mb_s: float 

#obje klase inheritaju BaseModel - Pydantic provjerava odgovaraju li podaci toj strukturi

#klasa TelemetrySample definira jedno cijelo mjerenje
class TelemetrySample(BaseModel):
    message_type: Literal["telemetry"]
    timestamp: float
    source: str
    metrics: Metrics

#----------------------------------------------

#DEFINICIJA EVENTA:

#workloadParameters - SAZNAT STO JOS TREBA UPISATI
class WorkloadParameters(BaseModel):
    duration: float

class Event(BaseModel):
    message_type: Literal["event"]
    timestamp: float
    type: str
    workload: str
    parameters: WorkloadParameters

#-------------------------------------------------

#BAZA PODATAKA:

DB_PATH = Path(__file__).with_name("telemetry.db")


def spremi_mjerenje(mjerenje: TelemetrySample):
    with closing(sqlite3.connect(DB_PATH)) as baza: 
        with baza:
            baza.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_samples (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    source TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                )
            """)
            baza.execute(
                """
                INSERT INTO telemetry_samples (timestamp, source, metrics_json)
                VALUES (?, ?, ?)
                """,
                (
                    mjerenje.timestamp,
                    mjerenje.source,
                    mjerenje.metrics.model_dump_json(),
                ),
            )

#-----------------------------------------------------------------------------------------------
app = FastAPI()


client = None #trenutno radim samo za jednog primatelja, pamtim samo njegovu jednu vezu

#clients: list[WebSocket] = [] #lista za pamcenje spojenih klijenata TREBA LI MI??


#HTTP GET ZA DOHVACANJE TELEMETRIJE UNUTAR NEKOG INTERVALA
@app.get("/telemetry")
def dohvati_telemetriju(start: float, end: float):
    if start>end:
        raise HTTPException(
            status_code=400,
            detail="start mora biti manji ili jednak end.",
        )

    with closing(sqlite3.connect(DB_PATH)) as baza: #otvaram vezu s bazom, a with closing ju automatski zatvara kad je zavrseno
        redovi = baza.execute(
            """
            SELECT timestamp, source, metrics_json
            FROM telemetry_samples
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp, id
            """,
            (start, end),
        ).fetchall()

        #POZIV - http://127.0.0.1:8000/telemetry?start=0&end=2000000000

    return [
        {
            "message_type": "telemetry",
            "timestamp": timestamp,
            "source": source,
            "metrics": json.loads(metrics_json),
        }
        for timestamp, source, metrics_json in redovi
    ]

#----------------------------------------------------------------------------



#WebSocket endpoint za primanje podataka iz dijela 2:
@app.websocket("/ws/agent")
async def receive_data(websocket: WebSocket):
    global latest_sample
    await websocket.accept() #prihvati vezu klijenta

    try:
        while True:
            poruka = await websocket.receive_text()

            try:
                podaci = json.loads(poruka)
                #json.loads pretvara json tekst u odgovarajuci python tip
                #ocekujem JSON objekt, koji postaje rjecnik
                #ako je poslan drugi JSON tip, npr. niz, rezultat nece biti rjecnik
                
                #radim odgovarajucu provjeru:
                if not isinstance(podaci, dict):
                    print("Poruka mora biti JSON objekt.")

                #dalje provjeravam tip poruke prema predlozenom fieldu message_type iz klase
                elif podaci.get("message_type") == "telemetry":
                    mjerenje = TelemetrySample.model_validate(podaci)
                    await asyncio.to_thread(spremi_mjerenje, mjerenje)
                    latest_sample = mjerenje 
                    print(f"Primljeno mjerenje: {mjerenje}")

                elif podaci.get("message_type") == "event":
                    dogadaj = Event.model_validate(podaci)
                    print(f"Primljen događaj: {dogadaj}")

                else:
                    print("Nepoznat ili nedostaje message_type.")

            #exception koji se dize ako json nije valjan (npr. postoji key ali nedostaje value)
            except json.JSONDecodeError:
                print("Poruka nije valjani JSON.")

            #pydantic daje ValidationError ako podaci ne odgovaraju odabranom modelu
            #npr. poruka poslana kao Event sadrzi timestamp: "abc" 
            except ValidationError as greska:
                print(f"Neispravni podaci: {greska}")
        
    except WebSocketDisconnect:
        print("Klijent se odspojio")
#--------------------------------------------------------------

latest_sample: TelemetrySample | None = None

#WebSocket endpoint za slanje podataka u dio 4:
@app.websocket("/ws/stream")
async def connect_client(websocket: WebSocket):
    global client

    if client is not None: #ne dopustam otvaranje nove veze ako jedna vec postoji
        await websocket.close()
        return

    await websocket.accept()
    client = websocket
    print("Klijent je povezan.")

    try:
        while True:
            zahtjev = await websocket.receive_text()
            print("Primljen zahtjev:", repr(zahtjev))

            if zahtjev != "get_latest":
                await websocket.send_json({"error": "unknown_request"})
            elif latest_sample is None:
                await websocket.send_json({"error": "no_telemetry_yet"})
            else:
                await websocket.send_json(latest_sample.model_dump(mode="json"))

    except WebSocketDisconnect:
        pass

    finally:
        client = None
        print("Client disconnected.")

    
#--------------------------------------------------------------------





if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1", port = 8000)

