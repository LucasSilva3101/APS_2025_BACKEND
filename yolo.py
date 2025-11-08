import os
import time
import uuid
import base64
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# =====================================================
# Carrega .env
# =====================================================
load_dotenv()
MONGO_URI       = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB        = os.getenv("MONGO_DB", "visionway")
MONGO_COL       = os.getenv("MONGO_COL", "historico")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
DETECTOR        = os.getenv("DETECTOR", "owlv2").lower()  # só OWL-V2 agora
OV_CONF_THRESHOLD = float(os.getenv("OV_CONF_THRESHOLD", "0.25"))
OV_LABELS_STR   = os.getenv(
    "OV_LABELS",
    "plastic bag,soda can,beer can,aluminum can,plastic bottle,glass bottle,"
    "milk carton,coffee cup,disposable cup,styrofoam,food container,"
    "plastic container,plastic wrapper,straw,plastic cutlery,cardboard,"
    "cardboard box,pizza box,bag,packaging,paper,napkin,mask,cigarette butt,"
    "plastic ring,six pack ring,rope,fishing net"
)
OV_LABELS       = [s.strip() for s in OV_LABELS_STR.split(",") if s.strip()]

# =====================================================
# Dicionário de tradução EN → PT
# =====================================================
OV_LABELS_PT = {
    "plastic bag": "sacola plástica",
    "soda can": "latinha de refrigerante",
    "beer can": "latinha de cerveja",
    "aluminum can": "lata de alumínio",
    "plastic bottle": "garrafa plástica",
    "glass bottle": "garrafa de vidro",
    "milk carton": "caixa de leite",
    "coffee cup": "copo de café",
    "disposable cup": "copo descartável",
    "styrofoam": "isopor",
    "food container": "embalagem de comida",
    "plastic container": "pote plástico",
    "plastic wrapper": "embalagem plástica",
    "straw": "canudo",
    "plastic cutlery": "talher plástico",
    "cardboard": "papelão",
    "cardboard box": "caixa de papelão",
    "pizza box": "caixa de pizza",
    "bag": "sacola",
    "packaging": "embalagem",
    "paper": "papel",
    "napkin": "guardanapo",
    "mask": "máscara",
    "cigarette butt": "bituca de cigarro",
    "plastic ring": "anel plástico",
    "six pack ring": "anel de latinhas",
    "rope": "corda",
    "fishing net": "rede de pesca",
}

# =====================================================
# FastAPI & CORS
# =====================================================
app = FastAPI(title="VisionWay API", version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MongoDB Setup
# =====================================================
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
        raise RuntimeError("Coleção MongoDB não inicializada.")
    return col

# =====================================================
# Modelos Pydantic
# =====================================================
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

# =====================================================
# Utilitários
# =====================================================
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

# =====================================================
# Carrega OWL-V2
# =====================================================
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

print("[OWL-V2] Carregando modelo 'google/owlv2-base-patch16-ensemble'...")
processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
ov_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "google/owlv2-base-patch16-ensemble"
)
print("[OWL-V2] Pronto. Labels:", OV_LABELS)

def infer_owlv2(img_bgr: np.ndarray):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    texts = [OV_LABELS]

    inputs = processor(text=texts, images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = ov_model(**inputs)

    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=OV_CONF_THRESHOLD
    )

    dets: List[Detection] = []
    annotated = img_bgr.copy()

    for box, score, lab_idx in zip(
        results[0]["boxes"], results[0]["scores"], results[0]["labels"]
    ):
        conf = float(score.item())
        label_en = OV_LABELS[int(lab_idx.item())]
        label = OV_LABELS_PT.get(label_en, label_en)  # traduz para PT se disponível
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        dets.append(Detection(label=label, confidence=round(conf, 4),
                              bbox_xyxy=[x1, y1, x2, y2]))
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 160), 2)
        cv2.putText(
            annotated,
            f"{label} {conf:.2f}",
            (int(x1), max(0, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 160),
            2,
            cv2.LINE_AA,
        )
    return annotated, dets

# =====================================================
# Rotas principais
# =====================================================
@app.get("/health")
def health():
    return {"ok": True, "service": "visionway", "ts": datetime.utcnow().isoformat() + "Z"}

@app.get("/version")
def version():
    return {
        "api_version": app.version,
        "detector": "owlv2",
        "ov_labels": OV_LABELS_PT,
        "conf_threshold": OV_CONF_THRESHOLD,
        "ts": datetime.utcnow().isoformat() + "Z",
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Tipo de arquivo inválido. Envie uma imagem.")
    try:
        start = time.perf_counter()
        content = await file.read()
        img = np_from_upload(content)
        annotated, dets = await asyncio.to_thread(infer_owlv2, img)
        out_b64 = to_base64_jpeg(annotated)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        resp = PredictResponse(
            image=out_b64,
            detections=dets,
            count=len(dets),
            timestamp=datetime.utcnow().isoformat() + "Z",
            meta={
                "detector": "owlv2",
                "model": "google/owlv2-base-patch16-ensemble",
                "elapsed_ms": elapsed_ms,
            },
        )

        doc = {
            "_id": str(uuid.uuid4()),
            "image": resp.image,
            "detections": [d.model_dump() for d in dets],
            "count": resp.count,
            "timestamp": resp.timestamp,
            "meta": resp.meta,
        }
        await mongo_col().insert_one(doc)
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
    return {"ok": True, "cleared": True, "total_deleted": result.deleted_count}

# =====================================================
# Execução local
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("yolo:app", host="127.0.0.1", port=8000, reload=True)