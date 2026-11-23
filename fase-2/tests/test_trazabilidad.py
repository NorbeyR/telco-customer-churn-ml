"""Trazabilidad del commit (corrección de la evaluación de la Fase 2).

Dentro de Docker no existe la carpeta .git: el commit llega por la variable de entorno
GIT_COMMIT, que el Dockerfile debe declarar (ARG) y transmitir (ENV) para que
src/versioning.py la escriba en metadata.json.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.training import run_training

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_dockerfile_declara_arg_y_lo_transmite_a_env():
    if not (ROOT / "Dockerfile").is_file():
        pytest.skip("El Dockerfile no está dentro de la imagen (se prueba en el repositorio)")
    lineas = [l.strip() for l in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()]
    arg = [i for i, l in enumerate(lineas) if re.match(r"ARG\s+GIT_COMMIT(=.*)?$", l)]
    env = [i for i, l in enumerate(lineas) if re.match(r"ENV\s+GIT_COMMIT=\$\{?GIT_COMMIT\}?$", l)]
    assert arg, "El Dockerfile debe declarar ARG GIT_COMMIT"
    assert env, "El Dockerfile debe transmitirlo con ENV GIT_COMMIT=${GIT_COMMIT}"
    assert arg[0] < env[0], "ARG debe declararse antes de usarse en ENV"
    assert not any(l.startswith("CMD") and "train.py" in l and "--version" in l for l in lineas), \
        "El entrenamiento no debe ejecutarse al construir la imagen"


def test_metadata_registra_el_commit_recibido_por_variable_de_entorno(tmp_path, data_csv, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", COMMIT)                       # lo que hace Docker (ENV)
    res = run_training(data_csv, "Churn", tmp_path / "models", "v1", seed=42)
    meta = json.loads((tmp_path / "models" / "v1" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["codigo"]["commit"] == COMMIT
    assert res["metadata"]["codigo"]["commit"] == COMMIT


def test_sin_variable_ni_git_el_commit_queda_marcado_como_no_disponible(tmp_path, data_csv, monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.setattr("src.versioning.PROJECT_ROOT", tmp_path)   # carpeta sin repositorio Git
    res = run_training(data_csv, "Churn", tmp_path / "models", "v1", seed=42)
    assert res["metadata"]["codigo"]["commit"] == "no-disponible"


def test_train_por_linea_de_comandos_guarda_el_commit_de_la_variable(tmp_path, data_csv):
    env = {**os.environ, "GIT_COMMIT": COMMIT}
    r = subprocess.run([sys.executable, "train.py", "--data", str(data_csv), "--output-dir",
                        str(tmp_path / "m"), "--version", "v-cli"], cwd=ROOT, env=env,
                       text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "m" / "v-cli" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["codigo"]["commit"] == COMMIT


def test_gitattributes_fija_saltos_de_linea_lf_para_csv_y_scripts():
    """Evita que el SHA-256 de los datos (metadata.json) y los .sh cambien según el sistema operativo."""
    ga = ROOT.parent / ".gitattributes"
    if not ga.is_file():
        pytest.skip(".gitattributes no está disponible fuera del repositorio (p. ej. dentro de la imagen)")
    reglas = ga.read_text(encoding="utf-8")
    assert re.search(r"^\*\.csv\s+text\s+eol=lf", reglas, re.M)
    assert re.search(r"^\*\.sh\s+text\s+eol=lf", reglas, re.M)
