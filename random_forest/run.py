"""
random_forest/run.py — Etapa 5: Entrenamiento RandomForest para mantenimiento predictivo.

Adaptaciones clave vs. caso de crédito base:
  - scoring='recall': optimizamos recall porque falso negativo (falla no detectada)
    tiene costo mucho mayor que falso positivo (parada innecesaria).
  - class_weight='balanced': compensación del desbalanceo (~3.4% fallas)
  - Se registran recall, precision y AUC en MLflow

Ejecutar: python random_forest/run.py
"""
import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | TRAIN | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET = "falla"
FEATURES = ["temperatura_aire", "temperatura_proceso", "velocidad_rpm",
            "torque_nm", "desgaste_herramienta_min", "delta_temp", "potencia_w",
            "tipo_maquina_enc"]
ARTIFACTS = Path("artifacts")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",        default="data/maintenance_train.csv")
    parser.add_argument("--experiment_name",   default="predictive_maintenance")
    parser.add_argument("--mlflow_uri",        default=None)
    parser.add_argument("--model_name",        default="PredictiveMaintenanceModel")
    parser.add_argument("--n_estimators",      default="100,200,300")
    parser.add_argument("--max_depth",         default="5,8,10")
    parser.add_argument("--min_samples_split", default="2,5,10")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"{args.train_data} no encontrado. Ejecuta los pasos anteriores.")

    log.info("Cargando datos: %s", args.train_data)
    df = pd.read_csv(args.train_data)
    X = df[FEATURES]
    y = df[TARGET]

    log.info("Train: %d filas | Fallas: %d (%.2f%%)", len(df), y.sum(), y.mean() * 100)

    mlflow.set_tracking_uri(args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns"))
    mlflow.set_experiment(args.experiment_name)

    param_grid = {
        "n_estimators":      [int(x) for x in args.n_estimators.split(",")],
        "max_depth":         [int(x) for x in args.max_depth.split(",")],
        "min_samples_split": [int(x) for x in args.min_samples_split.split(",")],
    }
    n_comb = len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) * len(param_grid["min_samples_split"])
    log.info("GridSearchCV: %d combinaciones x 5 folds (scoring: recall)", n_comb)

    gs = GridSearchCV(
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        param_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="recall",      # prioridad: no perder ninguna falla
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X, y)

    log.info("Mejores parámetros: %s", gs.best_params_)
    log.info("Mejor Recall CV:    %.4f", gs.best_score_)

    with mlflow.start_run(run_name="rf_maintenance") as run:
        mlflow.log_params(gs.best_params_)
        mlflow.log_params({
            "scoring":       "recall",
            "class_weight":  "balanced",
            "tasa_falla_pct": round(y.mean() * 100, 2),
        })

        y_pred  = gs.predict(X)
        y_proba = gs.predict_proba(X)[:, 1]
        mlflow.log_metrics({
            "cv_recall_mean":  round(gs.best_score_, 4),
            "train_recall":    round(recall_score(y, y_pred), 4),
            "train_precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "train_f1":        round(f1_score(y, y_pred), 4),
            "train_rocauc":    round(roc_auc_score(y, y_proba), 4),
            "train_prauc":     round(average_precision_score(y, y_proba), 4),
        })

        mlflow.sklearn.log_model(
            gs.best_estimator_,
            artifact_path="maintenance_model",
            registered_model_name=args.model_name,
        )
        run_id = run.info.run_id
        log.info("MLflow Run ID: %s", run_id)

    with open(ARTIFACTS / "modelo_mantenimiento.pkl", "wb") as f:
        pickle.dump(gs.best_estimator_, f)
    with open(ARTIFACTS / "train_run_id.txt", "w") as f:
        f.write(run_id)

    log.info("Modelo guardado: artifacts/modelo_mantenimiento.pkl")
