"""
data/run.py — Etapa 1: Descarga del dataset AI4I 2020 Predictive Maintenance.

Fuente primaria : UCI Machine Learning Repository — descarga directa sin credenciales.
  Dataset: AI4I 2020 Predictive Maintenance Dataset
  URL: https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip
  10,000 muestras | ~339 fallas (~3.4%) | 6 features + 5 tipos de falla
  Referencia: Matzka, S. (2020). https://doi.org/10.24432/C5HS5C

Fuente fallback : datos sintéticos con estructura idéntica (flag --sintetico)

Features del dataset:
  UDI              — identificador único (se descarta)
  Product ID       — id del producto (se descarta)
  Type             — tipo de máquina: L (low), M (medium), H (high quality)
  Air temperature [K]      — temperatura del aire en Kelvin
  Process temperature [K]  — temperatura del proceso en Kelvin
  Rotational speed [rpm]   — velocidad rotacional en RPM
  Torque [Nm]              — torque en Newton-metros
  Tool wear [min]          — desgaste de la herramienta en minutos

Target principal:
  Machine failure — 0=operación normal, 1=falla (cualquier tipo)

Ejecutar:
  python data/run.py             # descarga real desde UCI
  python data/run.py --sintetico # datos sintéticos (rápido)
"""
import argparse
import io
import logging
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DOWNLOAD | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT = "data/maintenance_raw.csv"

# Columnas que nos interesan (renombradas para el pipeline)
RENAME_MAP = {
    "Type":                     "tipo_maquina",
    "Air temperature [K]":      "temperatura_aire",
    "Process temperature [K]":  "temperatura_proceso",
    "Rotational speed [rpm]":   "velocidad_rpm",
    "Torque [Nm]":              "torque_nm",
    "Tool wear [min]":          "desgaste_herramienta_min",
    "Machine failure":          "falla",
}

UCI_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
UCI_CSV_FALLBACK = "https://raw.githubusercontent.com/dsaks/ai4i-dataset/main/ai4i2020.csv"


def descargar_uci() -> pd.DataFrame:
    import requests
    log.info("Descargando AI4I 2020 desde UCI Repository...")
    log.info("URL: %s", UCI_URL)

    resp = requests.get(UCI_URL, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        log.info("Archivos en ZIP: %s", csv_files)
        if not csv_files:
            raise ValueError("No se encontró CSV en el ZIP descargado")
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f)

    log.info("Dataset descargado: %d filas x %d columnas", df.shape[0], df.shape[1])
    return _limpiar_columnas(df)


def _limpiar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    cols_a_mantener = list(RENAME_MAP.keys())
    df = df[[c for c in cols_a_mantener if c in df.columns]].copy()
    df.rename(columns=RENAME_MAP, inplace=True)

    tasa_falla = df["falla"].mean() * 100
    log.info("Tasa de falla: %.2f%% (%d fallas de %d muestras)",
             tasa_falla, df["falla"].sum(), len(df))
    return df


def generar_sintetico(n: int = 5_000) -> pd.DataFrame:
    """
    Genera datos sintéticos con rangos físicamente plausibles del AI4I dataset.
    Distribuciones calibradas según el paper original de Matzka (2020).
    """
    log.warning("Modo SINTÉTICO — solo para desarrollo/tests, NO apto para producción.")
    rng = np.random.default_rng(42)

    tipos = rng.choice(["L", "M", "H"], size=n, p=[0.60, 0.30, 0.10])

    temp_aire = rng.normal(loc=300.0, scale=2.0, size=n).clip(295, 305)
    temp_proceso = temp_aire + rng.normal(loc=10.0, scale=1.0, size=n).clip(8, 12)
    velocidad = rng.normal(loc=1500, scale=200, size=n).clip(1168, 2886).round(0)
    torque = rng.normal(loc=40, scale=10, size=n).clip(3.8, 76.6).round(1)
    desgaste = rng.uniform(0, 250, size=n).round(0)

    # Reglas de falla simplificadas basadas en el paper AI4I
    falla_calor = ((temp_proceso - temp_aire) < 8.6) & (velocidad * torque < 9000)
    falla_desgaste = desgaste > 200
    falla_torque = (torque < 4) | (torque > 70)
    falla_potencia = (velocidad * torque) < 3500
    falla = (falla_calor | falla_desgaste | falla_torque | falla_potencia).astype(int)

    df = pd.DataFrame({
        "tipo_maquina":           tipos,
        "temperatura_aire":       temp_aire.round(1),
        "temperatura_proceso":    temp_proceso.round(1),
        "velocidad_rpm":          velocidad.astype(int),
        "torque_nm":              torque,
        "desgaste_herramienta_min": desgaste.astype(int),
        "falla":                  falla,
    })

    log.info("Sintético generado: %d filas | Fallas: %d (%.2f%%)",
             n, falla.sum(), falla.mean() * 100)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--sintetico", action="store_true",
                        help="Genera datos sintéticos en lugar de descargar de UCI")
    parser.add_argument("--n-sintetico", type=int, default=5_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path("data").mkdir(exist_ok=True)

    if args.sintetico:
        df = generar_sintetico(n=args.n_sintetico)
    else:
        try:
            df = descargar_uci()
        except Exception as e:
            log.warning("UCI no disponible (%s). Usando datos sintéticos como fallback.", e)
            df = generar_sintetico()

    df.to_csv(args.output, index=False)
    log.info("Dataset guardado: %s (%d filas)", args.output, len(df))
    log.info("Siguiente etapa: python preprocess/run.py")
