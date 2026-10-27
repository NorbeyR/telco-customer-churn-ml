"""Excepciones propias de la aplicación.

Todas heredan de ``ChurnError``: los scripts (train.py / predict.py) las capturan,
muestran un mensaje comprensible y terminan con código de salida distinto de cero.
"""


class ChurnError(Exception):
    """Error previsible de la aplicación (datos, modelo o versión inválidos)."""


class DataFileError(ChurnError):
    """El archivo de datos no existe, está vacío o no se puede leer."""


class MissingColumnsError(ChurnError):
    """Faltan columnas obligatorias en los datos de entrada."""


class InvalidValueError(ChurnError):
    """Hay valores con tipo o contenido incompatible con el modelo."""


class ModelNotFoundError(ChurnError):
    """El artefacto del modelo no existe o no se puede cargar."""


class VersionExistsError(ChurnError):
    """El identificador de versión ya existe (no se sobrescribe)."""


class InvalidVersionError(ChurnError):
    """El identificador de versión no es válido."""
