import pytest
from pathlib import Path
from src.versioning import (
    ensure_version_available,
    validate_version_id,
    VersionExistsError,
    InvalidVersionError,
    METADATA_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME
)

def test_validacion_id_version():
    """Verifica que la validación del identificador de versión funcione correctamente."""
    validate_version_id("v1")
    validate_version_id("v2_test")

    with pytest.raises(InvalidVersionError):
        validate_version_id("")


def test_no_sobrescribir_version_existente(tmp_path):
    """Verifica que intentar usar una versión existente genere VersionExistsError."""
    models_dir = tmp_path / "models"
    v1_dir = models_dir / "v1"
    v1_dir.mkdir(parents=True)

    # ensure_version_available recibe primero output_dir y luego version
    with pytest.raises(VersionExistsError):
        ensure_version_available(str(models_dir), "v1")


def test_version_disponible_ok(tmp_path):
    """Verifica que para un identificador nuevo no se genere error de versión."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    # ensure_version_available recibe primero output_dir y luego version
    ensure_version_available(str(models_dir), "v_nueva")