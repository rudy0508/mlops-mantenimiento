"""
evaluate/run.py — Etapa 6: Evaluación en test set + quality gate para mantenimiento predictivo.

Quality gate: Recall >= 0.85 AND AUC >= 0.85

Razonamiento del umbral:
  Un falso negativo (falla no detectada) puede resultar en:
  - Daño catastrófico a maquinaria ($10k-$1M+)
  - Parada no planificada de producción
  - Riesgo de seguridad para operadores
  Por eso priorizamos recall alto aunque sacrifiquemos precisión.

Ejecutar: python evaluate/run.py
"""
import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    average_precision_score, accuracy_score,
    classification_report, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | EVALUATE | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET = "falla"
FEATURES = ["temperatura_aire", "temperatura_proceso", "velocidad_rpm",
            "torque_nm", "desgaste_herramienta_min", "delta_temp", "potencia_w",
            "tipo_maquina_enc"]
ARTIFACTS = Path("artifacts")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",        default="data/maintenance_test.csv")
    parser.add_argument("--model_path",       default="artifacts/modelo_mantenimiento.pkl")
    parser.add_argument("--experiment_name",  default="predictive_maintenance")
    parser.add_argument("--mlflow_uri",       default=None)
    parser.add_argument("--recall_threshold", type=float, default=0.85)
    parser.add_argument("--auc_threshold",    type=float, default=0.85)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for p in [args.test_data, args.model_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos anteriores.")

    with open(args.model_path, "rb") as f:
        modelo = pickle.load(f)

    df_test = pd.read_csv(args.test_data)
    feats_disponibles = [f for f in FEATURES if f in df_test.columns]
    X_test = df_test[feats_disponibles]
    y_test = df_test[TARGET]

    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "test_recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "test_precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "test_f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "test_rocauc":    round(roc_auc_score(y_test, y_proba), 4),
        "test_prauc":     round(average_precision_score(y_test, y_proba), 4),
        "test_accuracy":  round(accuracy_score(y_test, y_pred), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 55)
    print(" MÉTRICAS — Mantenimiento Predictivo")
    print("=" * 55)
    for k, v in metricas.items():
        print(f"  {k:<20}: {v:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    Verdaderos Negativos (normal OK)     : {tn}")
    print(f"    Falsos Positivos (falsa alarma)      : {fp}")
    print(f"    Falsos Negativos (fallas perdidas!)  : {fn}  ← costo alto")
    print(f"    Verdaderos Positivos (fallas OK)     : {tp}")
    print(f"\n  Fallas detectadas: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
    print("\n" + classification_report(y_test, y_pred, target_names=["Normal", "Falla"]))

    mlflow.set_tracking_uri(args.mlflow_uri or "sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="evaluate_maintenance"):
        mlflow.log_metrics(metricas)
        mlflow.log_metrics({"test_false_negatives": int(fn), "test_false_positives": int(fp)})
        mlflow.log_param("recall_threshold", args.recall_threshold)
        mlflow.log_param("auc_threshold", args.auc_threshold)

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "eval_metrics.json", "w") as f:
        json.dump(metricas, f, indent=2)

    # Quality Gate
    print("\n" + "=" * 55)
    print(" QUALITY GATE")
    print("=" * 55)
    print(f"  Recall : {metricas['test_recall']:.4f} (umbral: >= {args.recall_threshold})")
    print(f"  ROC-AUC: {metricas['test_rocauc']:.4f} (umbral: >= {args.auc_threshold})")

    fallos = []
    if metricas["test_recall"] < args.recall_threshold:
        fallos.append(f"Recall {metricas['test_recall']:.4f} < {args.recall_threshold}")
    if metricas["test_rocauc"] < args.auc_threshold:
        fallos.append(f"ROC-AUC {metricas['test_rocauc']:.4f} < {args.auc_threshold}")

    if fallos:
        log.error("QUALITY GATE FALLIDO: %s", " | ".join(fallos))
        sys.exit(1)

    print("\n  APROBADO — modelo listo para despliegue")
