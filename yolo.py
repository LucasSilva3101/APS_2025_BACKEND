from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import base64

app = FastAPI()

# Libera o front local (ajuste origens se precisar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, restrinja!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo uma vez
model = YOLO("yolov8n-seg.pt")  # baixa na primeira execução

def np_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem.")
    return img

def run_inference_people_seg(img: np.ndarray):
    """Retorna imagem com fundo preto (apenas pessoas) e metadados das detecções."""
    results = model(img)
    output = np.zeros_like(img)
    dets = []  # lista de {label, conf, bbox}

    for result in results:
        if result.masks is None:
            continue

        # cada box corresponde a uma mask na mesma ordem
        for box, mask in zip(result.boxes, result.masks.data):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            # 0 = person (COCO)
            if cls_id == 0:
                # redimensiona máscara p/ imagem original
                mask_resized = cv2.resize(mask.cpu().numpy(), (img.shape[1], img.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # aplica máscara: copia pixels da pessoa para o output
                for c in range(3):
                    output[:, :, c] = np.where(mask_binary == 1, img[:, :, c], output[:, :, c])

                # coleta bbox (xyxy)
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = map(float, xyxy)
                dets.append({
                    "label": "person",
                    "confidence": round(conf, 4),
                    "bbox_xyxy": [x1, y1, x2, y2]
                })

    return output, dets

def to_base64_jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("Falha ao codificar a imagem de saída.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = np_from_upload(content)
        output, dets = run_inference_people_seg(img)
        out_b64 = to_base64_jpeg(output)
        return {
            "image": out_b64,         # imagem final (pessoas sobre fundo preto)
            "detections": dets,       # metadados (conf & bbox)
            "count": len(dets)
        }
    except Exception as e:
        return {"error": str(e)}
