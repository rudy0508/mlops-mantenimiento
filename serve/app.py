"""serve/app.py — API REST de predicción de fallas de maquinaria."""
import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | API-MANT | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURES = ["temperatura_aire", "temperatura_proceso", "velocidad_rpm",
            "torque_nm", "desgaste_herramienta_min", "delta_temp", "potencia_w",
            "tipo_maquina_enc"]
MODEL_PATH = Path("artifacts/modelo_mantenimiento.pkl")
UMBRAL_FALLA = float(os.getenv("UMBRAL_FALLA", "0.40"))  # más bajo: preferimos falsa alarma a falla perdida
TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}
modelo = None

# Rangos para normalización (deben coincidir con preprocess/run.py)
NORM_RANGES = {
    "temperatura_aire":         (295.0, 305.0),
    "temperatura_proceso":      (305.0, 314.0),
    "velocidad_rpm":            (1000, 3000),
    "torque_nm":                (1.0, 80.0),
    "desgaste_herramienta_min": (0, 300),
}


class LecturasSensor(BaseModel):
    """Lecturas de sensores de la máquina en tiempo real."""
    tipo_maquina:           str   = Field(..., description="Tipo: L (low), M (medium), H (high)")
    temperatura_aire:       float = Field(..., ge=290, le=320, description="Temperatura aire en Kelvin")
    temperatura_proceso:    float = Field(..., ge=295, le=325, description="Temperatura proceso en Kelvin")
    velocidad_rpm:          int   = Field(..., ge=0, le=5000, description="Velocidad rotacional en RPM")
    torque_nm:              float = Field(..., ge=0, le=100, description="Torque en Newton-metros")
    desgaste_herramienta_min: int = Field(..., ge=0, le=300, description="Desgaste de herramienta en minutos")

    model_config = {"json_schema_extra": {"example": {
        "tipo_maquina": "M",
        "temperatura_aire": 298.1,
        "temperatura_proceso": 308.6,
        "velocidad_rpm": 1551,
        "torque_nm": 42.8,
        "desgaste_herramienta_min": 0
    }}}


class PrediccionFalla(BaseModel):
    probabilidad_falla: float
    decision:           str     # FALLA_INMINENTE | OPERACION_NORMAL
    nivel_alerta:       str     # CRITICO | ADVERTENCIA | NORMAL
    score:              float
    umbral_usado:       float
    modelo:             str
    recomendacion:      str


class HealthResponse(BaseModel):
    status:  str
    modelo:  str
    version: str


def _normalizar(valor: float, col: str) -> float:
    lo, hi = NORM_RANGES[col]
    return max(0.0, min(1.0, (valor - lo) / (hi - lo)))


def _transformar_features(s: LecturasSensor) -> pd.DataFrame:
    """Aplica las mismas transformaciones que preprocess/run.py."""
    temp_aire_n     = _normalizar(s.temperatura_aire, "temperatura_aire")
    temp_proc_n     = _normalizar(s.temperatura_proceso, "temperatura_proceso")
    vel_n           = _normalizar(s.velocidad_rpm, "velocidad_rpm")
    torque_n        = _normalizar(s.torque_nm, "torque_nm")
    desgaste_n      = _normalizar(s.desgaste_herramienta_min, "desgaste_herramienta_min")

    delta_temp_raw  = s.temperatura_proceso - s.temperatura_aire
    potencia_raw    = s.torque_nm * s.velocidad_rpm * (2 * np.pi / 60)

    lo_d, hi_d = 8.0, 12.0
    delta_temp_n = max(0.0, min(1.0, (delta_temp_raw - lo_d) / (hi_d - lo_d)))

    lo_p, hi_p = 400.0, 25000.0
    potencia_n = max(0.0, min(1.0, (potencia_raw - lo_p) / (hi_p - lo_p)))

    tipo_enc = TYPE_ENCODING.get(s.tipo_maquina.upper(), 1)

    return pd.DataFrame([{
        "temperatura_aire":         temp_aire_n,
        "temperatura_proceso":      temp_proc_n,
        "velocidad_rpm":            vel_n,
        "torque_nm":                torque_n,
        "desgaste_herramienta_min": desgaste_n,
        "delta_temp":               delta_temp_n,
        "potencia_w":               potencia_n,
        "tipo_maquina_enc":         tipo_enc,
    }])


def _nivel_alerta(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "CRITICO", "Detener máquina para inspección inmediata"
    elif prob >= UMBRAL_FALLA:
        return "ADVERTENCIA", "Programar mantenimiento preventivo en las próximas horas"
    return "NORMAL", "Operación dentro de parámetros normales"


def cargar_modelo():
    global modelo
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            modelo = pickle.load(f)
        log.info("Modelo cargado: %s (%s)", MODEL_PATH, type(modelo).__name__)
    else:
        raise FileNotFoundError("Ejecuta: python random_forest/run.py")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelo()
    yield


app = FastAPI(
    title="API Mantenimiento Predictivo — MLOps Demo",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", tags=["Info"])
def root():
    return {"api": "Mantenimiento Predictivo", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Salud"])
def health():
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return HealthResponse(status="ok", modelo=type(modelo).__name__, version="1.0.0")


@app.post("/predict", response_model=PrediccionFalla, tags=["Prediccion"])
def predict(solicitud: LecturasSensor):
    """
    Evalúa lecturas de sensores y predice probabilidad de falla inminente.
    nivel_alerta: CRITICO | ADVERTENCIA | NORMAL
    """
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        df = _transformar_features(solicitud)
        feats = [f for f in FEATURES if f in df.columns]
        prob = float(modelo.predict_proba(df[feats])[0][1])
        nivel, recomendacion = _nivel_alerta(prob)
        return PrediccionFalla(
            probabilidad_falla=round(prob, 4),
            decision="FALLA_INMINENTE" if prob >= UMBRAL_FALLA else "OPERACION_NORMAL",
            nivel_alerta=nivel,
            score=round(prob, 4),
            umbral_usado=UMBRAL_FALLA,
            modelo=type(modelo).__name__,
            recomendacion=recomendacion,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
