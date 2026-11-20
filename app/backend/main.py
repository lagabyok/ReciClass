from __future__ import annotations

import base64
import binascii
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from feature_extractor import extraer_caracteristicas_desde_bytes

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "modelo" / "modelo.joblib"
FRONTEND_DIR = BASE_DIR / "frontend"

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class PredictionRequest(BaseModel):
    image_base64: str = Field(
        ...,
        min_length=20,
        description="Imagen codificada en Base64, sin el prefijo data:image/...;base64,"
    )
    filename: str | None = Field(default=None, max_length=255)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez cuando arranca el servidor."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontró el modelo en: {MODEL_PATH}. "
            "Copiá modelo.joblib en app/modelo/."
        )

    app.state.model = joblib.load(MODEL_PATH)
    yield
    app.state.model = None


app = FastAPI(
    title="ReciClass API",
    description="Clasificador de imágenes de residuos basado en RealWaste.",
    version="1.0.0",
    lifespan=lifespan,
)

# En producción, configurar FRONTEND_ORIGINS con el dominio del frontend.
# Ejemplo: FRONTEND_ORIGINS=https://mi-proyecto.netlify.app
origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5501,http://localhost:5501,http://127.0.0.7:5500"    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)





@app.get("/health")
def health():
    model_loaded = getattr(app.state, "model", None) is not None
    if not model_loaded:
        raise HTTPException(status_code=503, detail="El modelo no está cargado.")
    return {"estado": "ok", "modelo_cargado": True}


@app.post("/predecir")
def predecir(request: PredictionRequest):
    # Acepta Base64 puro. También tolera accidentalmente el prefijo data URL.
    encoded = request.image_base64.strip()
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="La imagen no contiene un Base64 válido.")

    if not image_bytes:
        raise HTTPException(status_code=422, detail="La imagen está vacía.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera el límite de 10 MB.")

    try:
        # Verificamos formato y decodificación antes de extraer características.
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=422,
                    detail="Formato no admitido. Usá JPG, PNG o WEBP."
                )
            image.verify()

        features = extraer_caracteristicas_desde_bytes(image_bytes)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"No se pudo procesar la imagen: {exc}")

    model = app.state.model
    try:
        # El notebook entrenó con vectores de 8366 características.
        if features.shape[1] != 8366:
            raise ValueError(
                f"Se generaron {features.shape[1]} características; se esperaban 8366."
            )

        prediction = str(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        # En SVC, classes_ mantiene el orden correspondiente a predict_proba.
        class_names = [str(name) for name in model.classes_]
        probability_map = {
            name: round(float(prob), 6)
            for name, prob in zip(class_names, probabilities)
        }
        confidence = float(np.max(probabilities))

        return {
            "prediccion": prediction,
            "probabilidad": round(confidence, 6),
            "probabilidades": probability_map,
            "archivo": request.filename,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {exc}")


# Servir el frontend desde la raíz de la aplicación.
app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend"
)