"""Versionamiento de modelos por carpetas: models/<version>/{modelo.joblib,metrics.json,metadata.json}."""
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import joblib
import sklearn

from .config import METADATA_FILENAME, METRICS_FILENAME, MODEL_FILENAME
from .errors import (InvalidVersionError, ModelNotFoundError, VersionExistsError)

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def validate_version_id(version: str) -> str:
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise InvalidVersionError(
            f"Identificador de versión inválido: {version!r}. Use letras, números, '.', '_' o '-' "
            "(máx. 64 caracteres), p. ej. v1, v2 o 2026-10-30_1530."
        )
    return version


def ensure_version_available(output_dir, version: str) -> Path:
    """Falla de inmediato si la versión ya existe (nunca se sobrescribe)."""
    validate_version_id(version)
    target = Path(output_dir) / version
    if target.exists():
        raise VersionExistsError(
            f"La versión '{version}' ya existe en '{output_dir}'. No se sobrescribe: "
            "use otro identificador (p. ej. la siguiente versión)."
        )
    return target


def save_version(output_dir, version: str, bundle: dict, metrics: dict, metadata: dict) -> dict:
    """Guarda la versión de forma atómica (carpeta temporal + renombrado).

    Si algo falla no queda una versión a medias; si el identificador ya existe, se rechaza.
    """
    target = ensure_version_available(output_dir, version)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{version}.tmp-", dir=out))
    try:
        joblib.dump(bundle, tmp / MODEL_FILENAME)
        for name, payload in ((METRICS_FILENAME, metrics), (METADATA_FILENAME, metadata)):
            with open(tmp / name, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
        if target.exists():                     # carrera improbable: otra ejecución la creó
            raise VersionExistsError(f"La versión '{version}' ya existe.")
        os.rename(tmp, target)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "directorio": str(target),
        "modelo": str(target / MODEL_FILENAME),
        "metricas": str(target / METRICS_FILENAME),
        "metadatos": str(target / METADATA_FILENAME),
    }


def resolve_model_path(model) -> Path:
    """Acepta la ruta del .joblib o la carpeta de la versión."""
    p = Path(model)
    if p.is_dir():
        p = p / MODEL_FILENAME
    if not p.is_file():
        raise ModelNotFoundError(f"No se encontró el modelo en '{model}'.")
    return p


def load_bundle(model) -> dict:
    path = resolve_model_path(model)
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelNotFoundError(f"No se pudo cargar el modelo '{path}': {exc}") from exc
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ModelNotFoundError(
            f"'{path}' no es un artefacto de esta aplicación (falta la clave 'pipeline')."
        )
    return bundle


def git_info() -> dict:
    """Commit del código usado. Dentro de Docker (sin .git) se toma de la variable GIT_COMMIT."""
    commit = os.environ.get("GIT_COMMIT")
    dirty = None
    try:
        if not commit:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
                                    capture_output=True, check=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=PROJECT_ROOT,
                                text=True, capture_output=True, check=True, timeout=10).stdout
        dirty = bool(status.strip())
    except Exception:
        pass
    return {"commit": commit or "no-disponible", "cambios_sin_commit": dirty}


def sklearn_version() -> str:
    return sklearn.__version__
