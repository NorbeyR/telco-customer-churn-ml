"""Registro persistente de las solicitudes a /predict (Fase 4).

Un CSV con una fila por intento, incluidas las rechazadas por validación:
    fecha_hora,estado_http,version,prediccion
    2026-11-10T15:02:11,200,v1,Churn
    2026-11-10T15:02:40,422,,
- fecha_hora: UTC, formato ISO sin zona (AAAA-MM-DDTHH:MM:SS).
- estado_http: código devuelto por la API.
- version / prediccion: solo en respuestas exitosas; vacíos en errores.
No se guardan los datos del cliente (privacidad): solo estos cuatro campos.
"""
import csv
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ("fecha_hora", "estado_http", "version", "prediccion")


class RequestLogger:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def log(self, estado_http: int, version: str = "", prediccion: str = "") -> None:
        """Agrega UNA fila (modo 'a': conserva el historial). Escribe el encabezado al crear el archivo.
        Un fallo al escribir el registro nunca debe tumbar la API: se avisa por stderr."""
        row = [datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), int(estado_http), version or "", prediccion or ""]
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                is_new = not self.path.exists() or self.path.stat().st_size == 0
                with open(self.path, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh, lineterminator="\n")
                    if is_new:
                        writer.writerow(FIELDS)
                    writer.writerow(row)
        except OSError as exc:
            print(f"AVISO: no se pudo escribir el registro '{self.path}': {exc}", file=sys.stderr)
