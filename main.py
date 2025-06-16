import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
import base64
import cv2

app = FastAPI()
session = ort.InferenceSession("converted_model.onnx")

class ImageData(BaseModel):
    image_base64: str

# Mapping des indices vers les noms de cancers
label_map = {
    0: "Pigmented Benign Keratosis",
    1: "Melanoma",
    2: "Vascular Lesion",
    3: "Actinic Keratosis",
    4: "Squamous Cell Carcinoma",
    5: "Basal Cell Carcinoma",
    6: "Seborrheic Keratosis",
    7: "Dermatofibroma",
    8: "Nevus"
}

@app.post("/predict")
async def predict(data: ImageData):
    # 1. Décoder l'image
    image_bytes = base64.b64decode(data.image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # 2. Redimensionner comme dans le training
    image = image.resize((100, 75))  # (width, height)
    # 3. Convertir en array float32 et normaliser [0,1]
    arr = np.array(image).astype(np.float32) / 255.0
    if arr.std() < 0.05:
        return {
            "prediction_index": -1,
            "prediction_label": "Image non valide ou trop uniforme",
            "probability": 0.0,
            "all_probabilities": {}
        }
    # Détection de flou
    gray = np.array(image.convert("L"))
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < 10:  # seuil à ajuster selon tes tests
        return {
            "prediction_index": -1,
            "prediction_label": "Image trop floue",
            "probability": 0.0,
            "all_probabilities": {}
        }
    arr = np.expand_dims(arr, axis=0)
    # 5. Inference ONNX
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})
    probs = outputs[0][0]  # (9,) pour 9 classes
    pred_idx = int(np.argmax(probs))
    pred_label = label_map[pred_idx]
    pred_prob = float(np.max(probs))
    all_probs = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    return {
        "prediction_index": pred_idx,
        "prediction_label": pred_label,
        "probability": pred_prob,
        "all_probabilities": all_probs
    }
