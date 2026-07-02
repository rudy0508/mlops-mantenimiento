"""
drift/run.py — Etapa 7: Detección de drift — Mantenimiento Predictivo.

En manufactura, el drift es gradual y predecible:
  - Desgaste progresivo de herramientas aumenta con el tiempo
  - Temperatura ambiente varía por estaciones
  - Velocidad y torque cambian cuando se procesan distintos materiales

Ejecutar: python drift/run.py
"""
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DRIFT | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURES = ["temperatura_aire", "temperatura_proceso", "velocidad_rpm",
            "torque_nm", "desgaste_herramienta_min", "delta_temp", "potencia_w",
            "tipo_maquina_enc"]
REPORTS = Path("reportes")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",       default="data/maintenance_train.csv")
    parser.add_argument("--current",         default="data/maintenance_test.csv")
    parser.add_argument("--drift_threshold", type=float, default=0.30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    REPORTS.mkdir(exist_ok=True)

    for p in [args.reference, args.current]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos anteriores.")

    feats_ref = [f for f in FEATURES if f in pd.read_csv(args.reference, nrows=1).columns]
    df_ref  = pd.read_csv(args.reference)[feats_ref]
    df_prod = pd.read_csv(args.current)[feats_ref]

    log.info("Referencia: %d | Actual: %d muestras", len(df_ref), len(df_prod))

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df_ref, current_data=df_prod)
        report.save_html(str(REPORTS / "drift_report.html"))
        resultado   = report.as_dict()
        drift_info  = resultado["metrics"][0]["result"]
        drift_det   = drift_info["dataset_drift"]
        drift_share = drift_info["share_of_drifted_columns"]
        drift_n     = drift_info["number_of_drifted_columns"]
        drift_total = drift_info["number_of_columns"]
    except ImportError:
        from scipy.stats import ks_2samp
        drifted     = sum(1 for c in feats_ref if ks_2samp(df_ref[c], df_prod[c]).pvalue < 0.05)
        drift_share = drifted / len(feats_ref)
        drift_det   = drift_share > args.drift_threshold
        drift_n, drift_total = drifted, len(feats_ref)

    resumen = {
        "drift_detectado":    drift_det,
        "features_con_drift": drift_n,
        "total_features":     drift_total,
        "share_drifted":      round(drift_share, 4),
        "umbral":             args.drift_threshold,
        "nota":               "Drift en desgaste_herramienta y temperaturas es esperable con el tiempo",
    }
    with open(REPORTS / "drift_summary.json", "w") as f:
        json.dump(resumen, f, indent=2)

    print(f"\n  Drift detectado    : {drift_det}")
    print(f"  Features con drift : {drift_n}/{drift_total} ({drift_share*100:.0f}%)")
    if drift_share > args.drift_threshold:
        log.warning("ALERTA: drift significativo en sensores — verificar calibración o re-entrenar")
    else:
        print("  Drift dentro de límites aceptables")
