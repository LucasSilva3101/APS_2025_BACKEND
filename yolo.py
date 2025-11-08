import os
import time
import uuid
import base64
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Path, Form
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv


# =========================
# Load .env
# =========================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "visionway")
MONGO_COL = os.getenv("MONGO_COL", "historico")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n-seg.pt")


# =========================
# App & CORS
# =========================
app = FastAPI(title="VisionWay API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MongoDB setup
# =========================
@app.on_event("startup")
async def startup_event():
    app.state.mongo_client = AsyncIOMotorClient(MONGO_URI)
    app.state.mongo_db = app.state.mongo_client[MONGO_DB]
    app.state.mongo_col = app.state.mongo_db[MONGO_COL]
    try:
        await app.state.mongo_col.create_index("timestamp")
        print("[Mongo] Conectado e índice em 'timestamp' garantido.")
    except Exception as e:
        print(f"[Mongo] Aviso ao criar índice: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    client = getattr(app.state, "mongo_client", None)
    if client:
        client.close()
        print("[Mongo] Conexão encerrada.")

def mongo_col():
    col = getattr(app.state, "mongo_col", None)
    if col is None:
        raise RuntimeError("Coleção MongoDB não inicializada. Verifique eventos de startup.")
    return col


# =========================
# YOLO model (segmentação)
# =========================
model = YOLO(MODEL_PATH)


# =========================
# Pydantic models
# =========================
class Detection(BaseModel):
    label: str
    confidence: float
    bbox_xyxy: List[float]

class PredictResponse(BaseModel):
    image: str
    detections: List[Detection]
    count: int
    timestamp: str
    meta: Dict[str, Any]

class HistoryItem(PredictResponse):
    id: str

class HistoryList(BaseModel):
    items: List[HistoryItem]
    total: int


# =========================
# Utilidades
# =========================
def np_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem.")
    return img

def to_base64_jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("Falha ao codificar a imagem.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _run_inference_people_seg_sync(img: np.ndarray):
    results = model(img)
    output = np.zeros_like(img)
    dets: List[Detection] = []

    for result in results:
        if result.masks is None:
            continue

        for box, mask in zip(result.boxes, result.masks.data):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0:
                mask_resized = cv2.resize(mask.cpu().numpy(), (img.shape[1], img.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                for c in range(3):
                    output[:, :, c] = np.where(mask_binary == 1, img[:, :, c], output[:, :, c])
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = map(float, xyxy)
                dets.append(Detection(label="person", confidence=round(conf, 4), bbox_xyxy=[x1, y1, x2, y2]))
    return output, dets


# =========================
# Rotas principais
# =========================
@app.get("/health")
def health():
    return {"ok": True, "service": "visionway", "ts": datetime.utcnow().isoformat() + "Z"}

@app.get("/version")
def version():
    names = getattr(model.model, "names", {})
    return {"api_version": app.version, "model_path": MODEL_PATH, "classes": names, "ts": datetime.utcnow().isoformat() + "Z"}

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    save_q: Optional[str] = Query("true", description="Sempre salva no MongoDB")
):
    # converte corretamente o parâmetro save (mesmo que venha como string)
    save = str(save_q).lower() == "true"
    print(f"[API] /predict chamado | save={save}")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Tipo de arquivo inválido. Envie uma imagem.")

    try:
        start = time.perf_counter()
        content = await file.read()
        img = np_from_upload(content)

        output, dets = await asyncio.to_thread(_run_inference_people_seg_sync, img)
        out_b64 = to_base64_jpeg(output)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        resp = PredictResponse(
            image=out_b64,
            detections=dets,
            count=len(dets),
            timestamp=datetime.utcnow().isoformat() + "Z",
            meta={"model": MODEL_PATH, "elapsed_ms": elapsed_ms}
        )

        if save:
            doc_id = str(uuid.uuid4())
            doc = {
                "_id": doc_id,
                "image": resp.image,
                "detections": [d.model_dump() for d in dets],
                "count": resp.count,
                "timestamp": resp.timestamp,
                "meta": resp.meta,
            }
            result = await mongo_col().insert_one(doc)
            print(f"[Mongo] Inserido _id={result.inserted_id}")
            resp.meta["history_id"] = doc_id

        return resp

    except Exception as e:
        print(f"[API] Erro no /predict: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/history", response_model=HistoryList)
async def get_history(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    col = mongo_col()
    total = await col.count_documents({})
    cursor = col.find({}).sort("timestamp", -1).skip(offset).limit(limit)

    items: List[HistoryItem] = []
    async for d in cursor:
        detections = [Detection(**det) for det in d.get("detections", [])]
        items.append(
            HistoryItem(
                id=str(d.get("_id")),
                image=d.get("image", ""),
                detections=detections,
                count=int(d.get("count", 0)),
                timestamp=d.get("timestamp", ""),
                meta=d.get("meta", {}),
            )
        )
    return HistoryList(items=items, total=total)

@app.delete("/history")
async def clear_history():
    result = await mongo_col().delete_many({})
    print(f"[Mongo] delete_many => {result.deleted_count} documentos removidos.")
    return {"ok": True, "cleared": True, "total_deleted": result.deleted_count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("yolo:app", host="0.0.0.0", port=8000, reload=True)
