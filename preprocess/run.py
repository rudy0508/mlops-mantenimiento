"""
preprocess/run.py — Etapa 2: Limpieza y transformación del dataset AI4I Predictive Maintenance.

Operaciones:
  - Encoding de tipo de máquina: L→0, M→1, H→2 (ordinal por calidad)
  - Temperatura diferencial: delta_temp = temp_proceso - temp_aire (feature derivada clave)
  - Potencia mecánica: potencia_w = velocidad_rpm * torque_nm * (2π/60) (feature física)
  - Imputación de nulos (mediana/moda)
  - Clip de outliers según rangos físicos del AI4I dataset
  - Normalización MinMaxScaler de features numéricas

Features derivadas: estas dos features son las más predictivas según el paper original
de Matzka (2020) ya que corresponden a los mecanismos de falla más comunes
(heat dissipation failure y power failure).

Ejecutar: python preprocess/run.py
"""
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | PREPROCESS | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET = "falla"
FEATURES_CAT = ["tipo_maquina"]
FEATURES_NUM_RAW = ["temperatura_aire", "temperatura_proceso", "velocidad_rpm", "torque_nm", "desgaste_herramienta_min"]
FEATURES_DERIVADAS = ["delta_temp", "potencia_w", "tipo_maquina_enc"]
FEATURES_FINALES = FEATURES_NUM_RAW + FEATURES_DERIVADAS + [TARGET]

# Rangos físicamente válidos del AI4I dataset
CLIP_RANGES = {
    "temperatura_aire":        (295.0, 305.0),
    "temperatura_proceso":     (305.0, 314.0),
    "velocidad_rpm":           (1000, 3000),
    "torque_nm":               (1.0, 80.0),
    "desgaste_herramienta_min": (0, 300),
}

TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}


def imputar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURES_NUM_RAW:
        n = df[col].isnull().sum()
        if n > 0:
            df[col].fillna(df[col].median(), inplace=True)
            log.info("  Imputados %d nulos en '%s' (mediana)", n, col)
    if "tipo_maquina" in df.columns:
        n = df["tipo_maquina"].isnull().sum()
        if n > 0:
            df["tipo_maquina"].fillna("M", inplace=True)
            log.info("  Imputados %d nulos en 'tipo_maquina' (moda: M)")
    return df


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in df.columns:
            n = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lo, hi)
            if n > 0:
                log.info("  Clipeados %d outliers en '%s' [%.1f, %.1f]", n, col, lo, hi)
    return df


def crear_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de ingeniería basadas en los mecanismos de falla del AI4I paper.
    """
    df = df.copy()

    # Diferencial de temperatura: clave para heat dissipation failure
    df["delta_temp"] = df["temperatura_proceso"] - df["temperatura_aire"]
    log.info("  Feature creada: delta_temp (rango: %.2f - %.2f)",
             df["delta_temp"].min(), df["delta_temp"].max())

    # Potencia mecánica en Watts: P = torque × velocidad_angular
    # omega = rpm × 2π/60
    df["potencia_w"] = df["torque_nm"] * df["velocidad_rpm"] * (2 * np.pi / 60)
    log.info("  Feature creada: potencia_w (rango: %.0f - %.0f W)",
             df["potencia_w"].min(), df["potencia_w"].max())

    # Encoding ordinal del tipo de máquina (L < M < H calidad)
    df["tipo_maquina_enc"] = df["tipo_maquina"].map(TYPE_ENCODING).fillna(1)
    log.info("  Encoding tipo_maquina: L=0, M=1, H=2")

    return df


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_num = FEATURES_NUM_RAW + ["delta_temp", "potencia_w"]
    scaler = MinMaxScaler()
    df[cols_num] = scaler.fit_transform(df[cols_num])
    log.info("  MinMaxScaler aplicado a: %s", cols_num)
    return df


def validar_schema(df: pd.DataFrame) -> None:
    errores = []
    if df.isnull().sum().sum() > 0:
        errores.append(f"NULOS RESIDUALES: {df.isnull().sum().sum()}")
    for col in FEATURES_FINALES:
        if col not in df.columns:
            errores.append(f"COLUMNA FALTANTE: {col}")
    if errores:
        raise ValueError("VALIDACIÓN FALLIDA:\n" + "\n".join(errores))
    log.info("  Schema OK — 0 nulos | Tasa de falla: %.2f%%", df[TARGET].mean() * 100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/maintenance_raw.csv")
    parser.add_argument("--output", default="data/maintenance_clean.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.input).exists():
        raise FileNotFoundError(f"{args.input} no encontrado. Ejecuta: python data/run.py")

    log.info("Cargando: %s", args.input)
    df = pd.read_csv(args.input)
    log.info("Shape inicial: %d x %d | Nulos: %d", df.shape[0], df.shape[1], df.isnull().sum().sum())

    df = imputar_nulos(df)
    df = clip_outliers(df)
    df = crear_features_derivadas(df)
    df = normalizar(df)
    validar_schema(df)

    # Mantener solo features finales
    df = df[FEATURES_FINALES]

    df.to_csv(args.output, index=False)
    log.info("Dataset limpio guardado: %s (%d filas)", args.output, len(df))
    log.info("Features: %s", FEATURES_FINALES)
