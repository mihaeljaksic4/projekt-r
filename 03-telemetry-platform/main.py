from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
import uvicorn
import json
from typing import Literal


#DEFINICIJA MJERENJA:

#klasa Metrics definira fieldove za izmjerene vrijednosti po dogovoru
class Metrics(BaseModel):
    cpu_usage: float | None = None
    gpu_usage: float | None = None
    ram_usage: float | None = None
    disk_io: float | None = None
    cpu_temp: float | None = None
    gpu_temp: float | None = None
    fan_rpm: float | None = None

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


app = FastAPI()


client = None #trenutno radim samo za jednog primatelja, pamtim samo njegovu jednu vezu

#clients: list[WebSocket] = [] #lista za pamcenje spojenih klijenata TREBA LI MI??

@app.get("/")
def root():
    return {"message": "Telemetry platform radi"}



#WebSocket endpoint za primanje podataka iz dijela 2:
@app.websocket("/ws/agent")
async def receive_data(websocket: WebSocket):
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
            await websocket.receive_text()
            #ne ocekujem zapravo tekst
            #koristim cinjenicu da recieve_text fja izaziva WebSocketDisconnect da bih prepoznala prekid veze

    except WebSocketDisconnect:
        pass

    finally:
        client = None
        print("Client disconnected.")

    
#--------------------------------------------------------------------





if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1", port = 8000)

