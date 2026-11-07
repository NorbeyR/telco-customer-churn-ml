"""Preprocesamiento: contrato de entrada + transformaciones aprendidas.

Corrección de la Fase 1: las conversiones de ``TotalCharges`` (espacios -> NaN -> número)
y ``SeniorCitizen`` (0/1 -> No/Yes) ahora viven DENTRO del Pipeline guardado, en el paso
``InputCleaner``. Así el artefacto acepta directamente el CSV original.

``InputCleaner`` NO aprende nada de los datos (solo aplica reglas fijas), por lo que no
puede producir fuga de información.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (CATEGORICAL_FEATURES, FLAG_01_COLUMNS, NULLABLE_NUMERIC,
                     NUMERIC_FEATURES)
from .errors import InvalidValueError, MissingColumnsError

MISSING_TOKEN = "__missing__"   # categoría para nulos en variables categóricas
_FLAG_MAP = {
    "0": "No", "0.0": "No", "no": "No", "false": "No",
    "1": "Yes", "1.0": "Yes", "yes": "Yes", "true": "Yes",
}


def _rows_text(mask) -> str:
    """Posiciones (desde 1, sin contar el encabezado) de las primeras filas afectadas."""
    pos = (np.flatnonzero(np.asarray(mask)) + 1).tolist()
    shown = ", ".join(str(p) for p in pos[:5])
    return shown + (f" ... (+{len(pos) - 5} más)" if len(pos) > 5 else "")


def _parse_numeric(s: pd.Series, col: str, allow_null: bool) -> pd.Series:
    raw = s.astype("string").str.strip()
    raw = raw.mask(raw == "")                      # espacios/vacío -> nulo
    num = pd.to_numeric(raw, errors="coerce")
    arr = num.to_numpy(dtype="float64", na_value=np.nan)

    not_number = raw.isna().to_numpy() == False  # noqa: E712  (tenía texto)
    bad_type = not_number & np.isnan(arr)
    bad_type |= np.isinf(arr)
    if bad_type.any():
        examples = raw[bad_type].head(3).tolist()
        raise InvalidValueError(
            f"La columna '{col}' debe ser numérica; valores no válidos (p. ej. {examples}) "
            f"en las filas: {_rows_text(bad_type)}."
        )
    negative = np.nan_to_num(arr, nan=0.0) < 0
    if negative.any():
        raise InvalidValueError(
            f"La columna '{col}' no admite valores negativos (filas: {_rows_text(negative)})."
        )
    if not allow_null and np.isnan(arr).any():
        raise InvalidValueError(
            f"La columna '{col}' no admite valores nulos "
            f"(filas: {_rows_text(np.isnan(arr))}). Solo se admiten nulos en {list(NULLABLE_NUMERIC)}."
        )
    return pd.Series(arr, index=s.index, name=col)


def _parse_flag(s: pd.Series, col: str) -> pd.Series:
    raw = s.astype("string").str.strip().str.lower()
    raw = raw.mask(raw == "")
    mapped = raw.map(_FLAG_MAP)
    unknown = raw.notna().to_numpy() & mapped.isna().to_numpy()
    if unknown.any():
        examples = s[unknown].head(3).tolist()
        raise InvalidValueError(
            f"La columna '{col}' solo admite 0/1 o No/Yes; valores no válidos (p. ej. "
            f"{examples}) en las filas: {_rows_text(unknown)}."
        )
    return mapped.astype("object").where(mapped.notna(), MISSING_TOKEN)


def _normalize_categorical(s: pd.Series) -> pd.Series:
    raw = s.astype("string").str.strip()
    raw = raw.mask(raw == "")
    return raw.astype("object").where(raw.notna(), MISSING_TOKEN)


class InputCleaner(BaseEstimator, TransformerMixin):
    """Valida el esquema de entrada y aplica las conversiones fijas de tipo.

    Contrato de entrada (documentado en el README):
      * Deben existir TODAS las columnas de ``numeric_features`` y ``categorical_features``;
        el orden de las columnas no importa; las columnas adicionales se ignoran.
      * ``tenure``, ``MonthlyCharges``: numéricas, sin nulos, >= 0.
      * ``TotalCharges``: numérica; vacío/espacios/nulo se admiten (-> NaN -> imputado con 0
        por el SimpleImputer, regla de la Fase 1 para clientes nuevos con tenure = 0).
      * ``SeniorCitizen``: 0/1 o No/Yes.
      * Categóricas: cualquier texto; un valor nulo se codifica como categoría desconocida.
    """

    def __init__(self, numeric_features=NUMERIC_FEATURES,
                 categorical_features=CATEGORICAL_FEATURES,
                 nullable_numeric=NULLABLE_NUMERIC,
                 flag_01_columns=FLAG_01_COLUMNS):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.nullable_numeric = nullable_numeric
        self.flag_01_columns = flag_01_columns

    @property
    def required_columns(self):
        return list(self.numeric_features) + list(self.categorical_features)

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise InvalidValueError("La entrada debe ser un DataFrame de pandas con nombres de columna.")
        if not X.columns.is_unique:
            dup = X.columns[X.columns.duplicated()].tolist()
            raise InvalidValueError(f"Hay columnas duplicadas en la entrada: {dup}")
        missing = [c for c in self.required_columns if c not in X.columns]
        if missing:
            raise MissingColumnsError(f"Faltan columnas obligatorias: {missing}")

    def fit(self, X, y=None):
        self._check_input(X)
        return self

    def transform(self, X):
        self._check_input(X)
        out = pd.DataFrame(index=X.index)
        for col in self.numeric_features:
            out[col] = _parse_numeric(X[col], col, allow_null=col in self.nullable_numeric)
        for col in self.categorical_features:
            if col in self.flag_01_columns:
                out[col] = _parse_flag(X[col], col)
            else:
                out[col] = _normalize_categorical(X[col])
        return out


def build_preprocessor(numeric_features=NUMERIC_FEATURES,
                       categorical_features=CATEGORICAL_FEATURES) -> ColumnTransformer:
    """Mismo diseño de la Fase 1: imputación 0 + StandardScaler y One-Hot."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", numeric, list(numeric_features)),
        ("cat", categorical, list(categorical_features)),
    ])


def build_pipeline(classifier) -> Pipeline:
    """InputCleaner -> ColumnTransformer -> clasificador (todo en un único artefacto)."""
    return Pipeline([
        ("cleaner", InputCleaner()),
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
