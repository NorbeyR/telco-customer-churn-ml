"""Flujo completo de entrenamiento (usado por train.py y por las pruebas)."""
import platform
from datetime import datetime, timezone
from pathlib import Path

from .config import (CATEGORICAL_FEATURES, DEFAULT_DATA_SOURCE, ID_COLUMN, NEGATIVE_LABEL,
                     NUMERIC_FEATURES, POSITIVE_LABEL)
from .data import load_labeled_data, sha256_of_file
from .evaluation import compute_metrics
from .modeling import (build_baseline, build_model, select_threshold_cv, split_train_test)
from .versioning import (ensure_version_available, git_info, save_version, sklearn_version)


def run_training(data_path, target, output_dir, version, seed=42, test_size=0.2, C=1.0,
                 max_iter=1000, optimize_threshold=False, cost_fn=5.0, cost_fp=1.0,
                 cv_folds=5, id_column=ID_COLUMN, data_source=DEFAULT_DATA_SOURCE) -> dict:
    """Entrena modelo + baseline, evalúa en prueba y guarda una versión nueva.

    Orden (evita fuga de información):
      1) separar train/test  2) ajustar TODO (transformaciones + clasificador) solo con train
      3) (opcional) elegir umbral con validación cruzada dentro de train  4) evaluar en test.
    """
    ensure_version_available(output_dir, version)          # falla rápido si ya existe

    X, y, groups = load_labeled_data(data_path, target, id_column)
    X_tr, X_te, y_tr, y_te, g_tr, strategy = split_train_test(X, y, test_size, seed, groups)

    baseline = build_baseline().fit(X_tr, y_tr)
    model = build_model(seed, C=C, max_iter=max_iter)

    threshold, threshold_info = 0.5, {"metodo": "fijo_0.5"}
    if optimize_threshold:
        threshold, threshold_info = select_threshold_cv(
            model, X_tr, y_tr, g_tr, seed, cv_folds, cost_fn, cost_fp)
    model.fit(X_tr, y_tr)

    p_model = model.predict_proba(X_te)[:, 1]
    p_base = baseline.predict_proba(X_te)[:, 1]
    metrics = {
        "metrica_principal": "f1_churn (con recall y roc_auc); evaluación sobre el conjunto de prueba",
        "etiquetas": {"1": f"{POSITIVE_LABEL} = abandona (Churn)", "0": f"{NEGATIVE_LABEL} = permanece"},
        "costos_supuestos": {"costo_abandono_no_detectado": cost_fn, "costo_falsa_alarma": cost_fp,
                             "unidad": "unidades relativas (supuesto del equipo, no cifras reales)"},
        "umbral_seleccion": threshold_info,
        "modelo": compute_metrics(y_te, p_model, threshold, cost_fn, cost_fp),
        "baseline": compute_metrics(y_te, p_base, 0.5, cost_fn, cost_fp),
    }
    if optimize_threshold:
        metrics["modelo_con_umbral_0.5"] = compute_metrics(y_te, p_model, 0.5, cost_fn, cost_fp)

    bundle = {
        "pipeline": model,
        "threshold": float(threshold),
        "target": target,
        "positive_label": POSITIVE_LABEL,
        "labels": {0: "No Churn", 1: "Churn"},
        "version": version,
        "sklearn_version": sklearn_version(),
    }
    metadata = {
        "version": version,
        "fecha_creacion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo_problema": "clasificación binaria",
        "variable_objetivo": target,
        "algoritmo": "LogisticRegression(class_weight='balanced') dentro de Pipeline "
                     "(InputCleaner -> ColumnTransformer -> clasificador)",
        "variables_entrada": {
            "numericas": list(NUMERIC_FEATURES),
            "categoricas": list(CATEGORICAL_FEATURES),
            "tipos_esperados": {
                "tenure": "numérico >= 0", "MonthlyCharges": "numérico >= 0",
                "TotalCharges": "numérico >= 0; vacío/espacios/nulo -> imputado con 0",
                "SeniorCitizen": "0/1 o No/Yes",
                "otras_categoricas": "texto; categorías no vistas se codifican en ceros",
            },
        },
        "parametros_modelo": {"C": C, "max_iter": max_iter, "class_weight": "balanced",
                              "penalizacion": "l2", "umbral_decision": threshold},
        "semilla": seed,
        "datos": {
            "fuente": data_source,
            "archivo": Path(data_path).name,
            "sha256": sha256_of_file(data_path),
            "n_observaciones": int(len(X)),
            "proporcion_churn": round(float(y.mean()), 4),
        },
        "particion": {"estrategia": strategy, "test_size": test_size,
                      "n_entrenamiento": int(len(X_tr)), "n_prueba": int(len(X_te)),
                      "metrica_evaluacion": "f1_churn, recall_churn, roc_auc, accuracy"},
        "entorno": {"python": platform.python_version(), "scikit-learn": sklearn_version(),
                    "dependencias": "requirements.txt"},
        "codigo": git_info(),
    }
    paths = save_version(output_dir, version, bundle, metrics, metadata)
    return {"metrics": metrics, "metadata": metadata, "paths": paths}
