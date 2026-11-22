#!/usr/bin/env python
"""Lee el registro de /predict y muestra el total de solicitudes y el porcentaje de fallos.

Ejemplo:
    python monitor.py --log logs/predicciones_log.csv
"""
import argparse
import os
import sys

from src.monitoring import compute_metrics, format_report, read_log

DEFAULT_LOG = os.environ.get("LOG_PATH", "logs/predicciones_log.csv")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Métricas de operación de la API a partir del registro de /predict.")
    p.add_argument("--log", default=DEFAULT_LOG, help="Archivo CSV del registro (por defecto: %(default)s).")
    args = p.parse_args(argv)
    try:
        rows = read_log(args.log)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"ERROR: no se pudo leer el registro '{args.log}': {exc}", file=sys.stderr)
        return 1
    print(format_report(compute_metrics(rows), args.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
