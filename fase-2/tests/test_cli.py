"""Línea de comandos: --help, ejecución completa y códigos de salida distintos de cero ante errores."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, text=True, capture_output=True)


def test_help_de_ambos_scripts():
    for script in ("train.py", "predict.py"):
        r = run(script, "--help")
        assert r.returncode == 0 and "--" in r.stdout


def test_entrenar_y_predecir_por_linea_de_comandos(tmp_path, data_csv, new_csv):
    r = run("train.py", "--data", str(data_csv), "--target", "Churn",
            "--output-dir", str(tmp_path / "models"), "--version", "v1", "--seed", "42")
    assert r.returncode == 0, r.stderr
    out_csv = tmp_path / "out" / "pred.csv"
    r = run("predict.py", "--model", str(tmp_path / "models" / "v1" / "modelo.joblib"),
            "--input", str(new_csv), "--output", str(out_csv))
    assert r.returncode == 0, r.stderr
    res = pd.read_csv(out_csv)
    assert len(res) == 12 and "prediccion" in res.columns


def test_columna_ausente_termina_con_codigo_distinto_de_cero_y_sin_archivo(tmp_path, trained, new_csv):
    out, _ = trained
    bad = tmp_path / "bad.csv"
    pd.read_csv(new_csv).drop(columns=["tenure"]).to_csv(bad, index=False)
    out_csv = tmp_path / "pred.csv"
    r = run("predict.py", "--model", str(out / "v1"), "--input", str(bad), "--output", str(out_csv))
    assert r.returncode != 0
    assert "tenure" in r.stderr
    assert not out_csv.exists()


def test_version_repetida_por_cli_falla(tmp_path, data_csv):
    args = ("train.py", "--data", str(data_csv), "--output-dir", str(tmp_path / "m"), "--version", "v1")
    assert run(*args).returncode == 0
    r = run(*args)
    assert r.returncode != 0 and "ya existe" in r.stderr