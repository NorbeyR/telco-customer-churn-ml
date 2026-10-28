"""Lectura y validación de datos etiquetados (entrenamiento)."""
import hashlib
from pathlib import Path

import pandas as pd

from .config import ID_COLUMN, NEGATIVE_LABEL, POSITIVE_LABEL, REQUIRED_FEATURES
from .errors import DataFileError, InvalidValueError, MissingColumnsError


def read_csv(path) -> pd.DataFrame:
    """Lee un CSV con mensajes de error comprensibles."""
    p = Path(path)
    if not p.is_file():
        raise DataFileError(f"No existe el archivo de datos: '{p}'.")
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        raise DataFileError(f"El archivo '{p}' está vacío.") from None
    except Exception as exc:  # CSV mal formado, codificación, etc.
        raise DataFileError(f"No se pudo leer '{p}' como CSV: {exc}") from exc
    if df.empty:
        raise DataFileError(f"El archivo '{p}' no contiene filas de datos.")
    return df


def sha256_of_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labeled_data(path, target: str, id_column: str = ID_COLUMN):
    """Devuelve (X, y, groups).

    - X: solo las variables predictoras requeridas (sin objetivo ni identificador).
    - y: 1 = Yes (abandona), 0 = No (permanece).
    - groups: identificador de cliente si hay clientes repetidos; si no, None.
    """
    df = read_csv(path)
    if target not in df.columns:
        raise MissingColumnsError(
            f"La variable objetivo '{target}' no está en el archivo. "
            f"Columnas encontradas: {list(df.columns)}"
        )
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise MissingColumnsError(f"Faltan columnas obligatorias en los datos: {missing}")

    raw_target = df[target].astype("string").str.strip()
    if raw_target.isna().any():
        n = int(raw_target.isna().sum())
        raise InvalidValueError(f"La variable objetivo '{target}' tiene {n} valores nulos.")
    invalid = sorted(set(raw_target) - {POSITIVE_LABEL, NEGATIVE_LABEL})
    if invalid:
        raise InvalidValueError(
            f"La variable objetivo '{target}' solo admite '{POSITIVE_LABEL}'/'{NEGATIVE_LABEL}'; "
            f"se encontraron: {invalid[:5]}"
        )
    y = (raw_target == POSITIVE_LABEL).astype(int).reset_index(drop=True)

    X = df.loc[:, list(REQUIRED_FEATURES)].reset_index(drop=True)

    groups = None
    if id_column in df.columns and df[id_column].duplicated().any():
        groups = df[id_column].reset_index(drop=True)
    return X, y, groups
