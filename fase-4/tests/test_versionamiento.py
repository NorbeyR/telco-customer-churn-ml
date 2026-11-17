"""Prueba 5 - Versionamiento: segunda versión conserva la primera; id repetido no sobrescribe."""
import hashlib

import pytest

from src.errors import InvalidVersionError, VersionExistsError
from src.training import run_training
from src.versioning import load_bundle


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_segunda_version_conserva_la_primera_y_ambas_cargan(tmp_path, data_csv):
    out = tmp_path / "models"
    run_training(data_csv, "Churn", out, "v1", seed=42)
    h1 = _digest(out / "v1" / "modelo.joblib")
    run_training(data_csv, "Churn", out, "v2", seed=7, optimize_threshold=True, cv_folds=3)
    assert sorted(p.name for p in out.iterdir()) == ["v1", "v2"]
    assert _digest(out / "v1" / "modelo.joblib") == h1
    b1, b2 = load_bundle(out / "v1"), load_bundle(out / "v2")
    assert b1["version"] == "v1" and b2["version"] == "v2"
    assert b1["threshold"] == 0.5
    assert 0.05 <= b2["threshold"] <= 0.95


def test_identificador_repetido_no_sobrescribe(tmp_path, data_csv):
    out = tmp_path / "models"
    run_training(data_csv, "Churn", out, "v1", seed=42)
    before = {p.name: _digest(p) for p in (out / "v1").iterdir()}
    with pytest.raises(VersionExistsError):
        run_training(data_csv, "Churn", out, "v1", seed=1)
    after = {p.name: _digest(p) for p in (out / "v1").iterdir()}
    assert before == after


@pytest.mark.parametrize("bad", ["", "../fuera", "v 1", "a/b"])
def test_identificador_invalido(tmp_path, data_csv, bad):
    with pytest.raises(InvalidVersionError):
        run_training(data_csv, "Churn", tmp_path / "m", bad)