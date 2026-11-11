"""API REST del modelo de churn (Fase 3).

Rutas:
  GET  /health    -> la API está en ejecución
  GET  /versions  -> versiones disponibles y versión activa
  POST /predict   -> predicción para UN registro JSON con las variables predictoras
  POST /train     -> entrena y guarda una versión nueva (no cambia la versión activa)

Reutiliza SIN cambios la lógica de la Fase 2 (src/): validación y preprocesamiento
guardados en el Pipeline, predict_dataframe() y run_training().

Configuración por variables de entorno (ver README):
  MODEL_VERSION (v1), MODELS_DIR (models), DATA_PATH, TARGET (Churn), SEED (42)
"""
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import ID_COLUMN, MODEL_FILENAME, TARGET
from src.errors import (ChurnError, DataFileError, InvalidValueError, InvalidVersionError,
                        MissingColumnsError, ModelNotFoundError, VersionExistsError)
from src.inference import predict_dataframe
from src.training import run_training
from src.versioning import load_bundle, validate_version_id

DEFAULT_DATA = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


@dataclass
class Settings:
    model_version: str = "v1"
    models_dir: str = "models"
    data_path: str = DEFAULT_DATA
    target: str = TARGET
    seed: int = 42

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_version=os.environ.get("MODEL_VERSION", "v1"),
            models_dir=os.environ.get("MODELS_DIR", "models"),
            data_path=os.environ.get("DATA_PATH", DEFAULT_DATA),
            target=os.environ.get("TARGET", TARGET),
            seed=int(os.environ.get("SEED", "42")),
        )


def list_versions(models_dir) -> list:
    d = Path(models_dir)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and not p.name.startswith(".") and (p / MODEL_FILENAME).is_file())


def next_version_id(models_dir) -> str:
    """Siguiente identificador libre con forma vN (v3 si ya existen v1 y v2)."""
    nums = [int(m.group(1)) for v in list_versions(models_dir) if (m := re.fullmatch(r"v(\d+)", v))]
    return f"v{max(nums, default=0) + 1}"


def _error(status: int, mensaje: str, **extra) -> JSONResponse:
    extra = {k: v for k, v in extra.items() if v}
    return JSONResponse(status_code=status, content={"detail": mensaje, **extra})


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    app = FastAPI(title="API de predicción de churn", version="3.0",
                  description="Fase 3 - Modelos y Simulación de Sistemas I")
    cache: dict = {}
    train_lock = threading.Lock()

    def get_bundle() -> dict:
        """Carga (y cachea) la versión configurada; 503 con instrucciones si no existe."""
        path = Path(cfg.models_dir) / cfg.model_version / MODEL_FILENAME
        try:
            validate_version_id(cfg.model_version)
        except InvalidVersionError as exc:
            raise ModelNotFoundError(str(exc)) from exc
        if not path.is_file():
            raise ModelNotFoundError(
                f"La versión configurada '{cfg.model_version}' no está disponible en "
                f"'{cfg.models_dir}'. Versiones disponibles: {list_versions(cfg.models_dir) or 'ninguna'}. "
                "Seleccione una con la variable de entorno MODEL_VERSION y reinicie la API.")
        key = (str(path), path.stat().st_mtime_ns)
        if key not in cache:
            cache.clear()
            cache[key] = load_bundle(path)
        return cache[key]

    # ---------------------------------------------------------------- errores
    @app.exception_handler(MissingColumnsError)
    async def _missing(_: Request, exc: MissingColumnsError):
        return _error(422, str(exc), campos_faltantes=exc.columns)

    @app.exception_handler(InvalidValueError)
    async def _invalid(_: Request, exc: InvalidValueError):
        return _error(422, str(exc))

    @app.exception_handler(ModelNotFoundError)
    async def _no_model(_: Request, exc: ModelNotFoundError):
        return _error(503, str(exc), versiones_disponibles=list_versions(cfg.models_dir))

    @app.exception_handler(VersionExistsError)
    async def _exists(_: Request, exc: VersionExistsError):
        return _error(409, str(exc))

    @app.exception_handler(InvalidVersionError)
    async def _bad_version(_: Request, exc: InvalidVersionError):
        return _error(422, str(exc))

    @app.exception_handler(DataFileError)
    async def _no_data(_: Request, exc: DataFileError):
        return _error(503, f"Datos de entrenamiento no disponibles: {exc}")

    @app.exception_handler(ChurnError)
    async def _churn(_: Request, exc: ChurnError):
        return _error(400, str(exc))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        return _error(500, f"Error interno ({type(exc).__name__}). Revise los registros del servidor.")

    # ------------------------------------------------------------------ rutas
    @app.get("/health")
    def health():
        versions = list_versions(cfg.models_dir)
        return {"status": "ok", "mensaje": "La API está en ejecución",
                "version_activa": cfg.model_version,
                "modelo_disponible": cfg.model_version in versions}

    @app.get("/versions")
    def versions():
        return {"version_activa": cfg.model_version, "versiones_disponibles": list_versions(cfg.models_dir)}

    @app.post("/predict")
    async def predict(request: Request):
        try:
            record = json.loads((await request.body()) or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(400, "El cuerpo de la solicitud no es un JSON válido.")
        if not isinstance(record, dict):
            return _error(400, "Se esperaba un objeto JSON con las variables predictoras "
                               '(p. ej. {"tenure": 12, "Contract": "One year", ...}).')
        bad = [k for k, v in record.items() if isinstance(v, (dict, list))]
        if bad:
            return _error(422, f"Los campos {bad} deben ser valores simples (número, texto o null), no objetos ni listas.")

        bundle = get_bundle()                                   # 503 si no hay modelo
        result, report = predict_dataframe(bundle, pd.DataFrame([record]))   # 422 si falta/inválido
        row = result.iloc[0]
        report["columnas_extra_ignoradas"] = [c for c in report["columnas_extra_ignoradas"] if c != ID_COLUMN]
        identificador = {ID_COLUMN: str(row[ID_COLUMN])} if ID_COLUMN in result.columns else {}
        return {
            **identificador,
            "prediccion": row["prediccion_etiqueta"],           # etiqueta: "Churn" / "No Churn"
            "clase": int(row["prediccion"]),                    # 1 = abandona, 0 = permanece
            "probabilidad_churn": float(row["probabilidad_churn"]),
            "umbral": result.attrs["umbral"],
            "version": bundle["version"],
            "advertencias": {k: v for k, v in report.items() if v},
        }

    @app.post("/train")
    async def train(request: Request):
        raw = await request.body()
        try:
            body = json.loads(raw) if raw.strip() else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(400, "El cuerpo de la solicitud no es un JSON válido (puede enviarse vacío).")
        if not isinstance(body, dict):
            return _error(400, "El cuerpo debe ser un objeto JSON o estar vacío.")
        unknown = sorted(set(body) - {"version", "seed", "optimize_threshold"})
        if unknown:
            return _error(422, f"Campos no admitidos: {unknown}. Admitidos: version, seed, optimize_threshold.")
        try:
            seed = int(body.get("seed", cfg.seed))
        except (TypeError, ValueError):
            return _error(422, "'seed' debe ser un entero.")
        optimize = bool(body.get("optimize_threshold", False))

        from starlette.concurrency import run_in_threadpool
        def job():
            with train_lock:                                     # un entrenamiento a la vez
                version = body.get("version") or next_version_id(cfg.models_dir)
                return run_training(cfg.data_path, cfg.target, cfg.models_dir, version, seed=seed,
                                    optimize_threshold=optimize)
        res = await run_in_threadpool(job)
        m = res["metrics"]["modelo"]
        return JSONResponse(status_code=201, content={
            "version": res["metadata"]["version"],
            "ruta": res["paths"]["directorio"],
            "metricas": {k: m[k] for k in ("accuracy", "recall_churn", "f1_churn", "roc_auc", "umbral")},
            "mensaje": "Versión guardada. La API sigue usando "
                       f"'{cfg.model_version}' hasta que cambie MODEL_VERSION y reinicie.",
        })

    return app


app = create_app()
