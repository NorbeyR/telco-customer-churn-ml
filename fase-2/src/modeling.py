"""Separación de datos, construcción de modelos y selección de umbral (solo con train)."""
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict, train_test_split)

from .errors import ChurnError
from .evaluation import total_cost
from .preprocessing import build_pipeline


def split_train_test(X, y, test_size: float, seed: int, groups=None):
    """Separa ANTES de ajustar cualquier transformación.

    Estratificada por la variable objetivo. Si hay clientes repetidos (``groups``),
    todas las filas de un mismo cliente quedan en el mismo conjunto.
    Devuelve (X_train, X_test, y_train, y_test, groups_train, estrategia).
    """
    if not 0 < test_size < 1:
        raise ChurnError("--test-size debe estar entre 0 y 1 (exclusivo).")
    if groups is None:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)
        return X_tr, X_te, y_tr, y_te, None, "estratificada por objetivo (train_test_split)"
    n_splits = max(2, round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, te_idx = next(sgkf.split(X, y, groups))
    return (X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx],
            groups.iloc[tr_idx], "estratificada por grupos de cliente (StratifiedGroupKFold)")


def build_model(seed: int, C: float = 1.0, max_iter: int = 1000):
    """Regresión logística balanceada (L2 por defecto: la regularización reparte el peso
    entre predictores correlacionados en lugar de inflar sus coeficientes)."""
    clf = LogisticRegression(class_weight="balanced", C=C, max_iter=max_iter, random_state=seed)
    return build_pipeline(clf)


def build_baseline():
    """Modelo base de la Fase 1: siempre predice la clase mayoritaria."""
    return build_pipeline(DummyClassifier(strategy="most_frequent"))


def select_threshold_cv(pipeline, X_train, y_train, groups_train, seed: int, folds: int,
                        cost_fn: float, cost_fp: float):
    """Elige el umbral que minimiza el costo con validación cruzada DENTRO del entrenamiento.

    El conjunto de prueba no interviene. Las transformaciones se reajustan en cada partición
    porque se clona el Pipeline completo.
    """
    if groups_train is None:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X_train, y_train))
    else:
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X_train, y_train, groups_train))
    proba = cross_val_predict(clone(pipeline), X_train, y_train, cv=splits,
                              method="predict_proba")[:, 1]
    y = np.asarray(y_train)

    def cost_at(t):
        pred = proba >= t
        fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum())
        return total_cost(fn, fp, cost_fn, cost_fp)

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = np.array([cost_at(t) for t in grid])
    best = costs.min()
    candidates = grid[costs == best]
    chosen = float(candidates[np.argmin(np.abs(candidates - 0.5))])   # empate: el más cercano a 0,5
    return chosen, {
        "metodo": "cv_costo_minimo",
        "cv_folds": folds,
        "costo_cv_umbral_0.5": cost_at(0.5),
        "costo_cv_umbral_elegido": float(best),
    }
