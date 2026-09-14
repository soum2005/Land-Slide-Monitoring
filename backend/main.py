import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from database import Base, SessionLocal, engine
from ml_engine import load_artifacts
from seed_data import seed

from routers import (
    admin,
    alerts,
    auth,
    dashboard,
    emergency,
    incidents,
    map_data,
    predictions,
    rainfall,
    reports,
    sensors,
    simulation,
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed(db)
    finally:
        db.close()
    load_artifacts()
    yield


app = FastAPI(
    title="Bhu-Surakha Disaster Intelligence Platform",
    version="2.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(map_data.router)
app.include_router(predictions.router)
app.include_router(rainfall.router)
app.include_router(sensors.router)
app.include_router(alerts.router)
app.include_router(incidents.router)
app.include_router(emergency.router)
app.include_router(reports.router)
app.include_router(simulation.router)
app.include_router(admin.router)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ner-landslide-ews"}


@app.websocket("/ws/telemetry")
async def telemetry_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            await ws.send_json({"echo": msg, "hint": "Use POST /api/simulation/tick for live risk ticks"})
    except WebSocketDisconnect:
        return
