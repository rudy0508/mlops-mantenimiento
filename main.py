"""
main.py — Orquestador del pipeline MLOps: Mantenimiento Predictivo de Maquinaria.

Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
         https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
         10,000 muestras | 339 fallas (~3.4%) | descarga directa sin credenciales
         Features: Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear
         Target: Machine failure (0=normal, 1=falla)

Uso:
    python main.py                              # todas las etapas
    python main.py --steps download preprocess  # etapas específicas
    python main.py --sintetico                  # datos sintéticos (dev rápido)
"""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | PIPELINE-MANT | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_run.log"),
    ],
)
log = logging.getLogger(__name__)

ALL_STEPS = ["download", "preprocess", "segregate", "check_data", "random_forest", "evaluate", "drift"]


def build_commands(sintetico: bool) -> dict:
    download_cmd = [sys.executable, "data/run.py"]
    if sintetico:
        download_cmd.append("--sintetico")
    return {
        "download":      download_cmd,
        "preprocess":    [sys.executable, "preprocess/run.py"],
        "segregate":     [sys.executable, "segregate/run.py"],
        "check_data":    [sys.executable, "-m", "pytest", "check_data/test_data.py", "-v", "--tb=short"],
        "random_forest": [sys.executable, "random_forest/run.py"],
        "evaluate":      [sys.executable, "evaluate/run.py"],
        "drift":         [sys.executable, "drift/run.py"],
    }


def ejecutar_paso(nombre: str, cmd: list) -> tuple[bool, float]:
    inicio = time.time()
    log.info(">>> Iniciando: %s", nombre)
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns")
    result = subprocess.run(cmd, capture_output=False, env=env)
    dur = round(time.time() - inicio, 2)
    ok = result.returncode == 0
    if ok:
        log.info("<<< Completado: %s (%.2f s)", nombre, dur)
    else:
        log.error("XXX FALLO: %s (código: %d)", nombre, result.returncode)
    return ok, dur


def main():
    parser = argparse.ArgumentParser(description="Pipeline MLOps — Mantenimiento Predictivo")
    parser.add_argument("--steps", nargs="+", default=ALL_STEPS, choices=ALL_STEPS)
    parser.add_argument("--sintetico", action="store_true",
                        help="Usa datos sintéticos (no requiere descarga de UCI)")
    args = parser.parse_args()

    for d in ["data", "artifacts", "reportes"]:
        Path(d).mkdir(exist_ok=True)

    step_commands = build_commands(args.sintetico)

    log.info("=" * 60)
    log.info(" PIPELINE MLOps — Mantenimiento Predictivo de Maquinaria")
    log.info(" Dataset: AI4I 2020 (UCI)")
    log.info(" Etapas: %s", " → ".join(args.steps))
    log.info("=" * 60)

    resumen = []
    for paso in args.steps:
        ok, dur = ejecutar_paso(paso, step_commands[paso])
        resumen.append({"paso": paso, "estado": "OK" if ok else "FALLO", "duracion_s": dur})
        if not ok:
            log.error("Pipeline detenido en: %s", paso)
            sys.exit(1)

    dur_total = sum(r["duracion_s"] for r in resumen)
    log.info("=" * 60)
    log.info(" PIPELINE COMPLETADO EN %.2f segundos", dur_total)
    log.info("=" * 60)
    for r in resumen:
        log.info("  [%s] %s (%.2f s)", r["estado"], r["paso"], r["duracion_s"])
    log.info("")
    log.info("  Siguiente: uvicorn serve.app:app --host 0.0.0.0 --port 8002")


if __name__ == "__main__":
    main()
