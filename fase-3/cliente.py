#!/usr/bin/env python
"""Cliente de prueba de la API (solo librería estándar: no necesita instalar nada).

Ejemplos (con la API en http://localhost:8000):
    python cliente.py health
    python cliente.py predict                      # primera fila de data/ejemplo_prediccion.csv
    python cliente.py predict --fila 2             # otra fila del CSV de ejemplo
    python cliente.py predict --sin-campo tenure   # entrada inválida: falta una variable obligatoria
    python cliente.py train                        # entrena y guarda una versión nueva (puede tardar unos segundos)
    python cliente.py versions
    python cliente.py demo                         # 2 predicciones válidas + 1 inválida (ensayo de monitoreo)
"""
import argparse
import csv
import json
import sys
import urllib.error
import urllib.request

INT_COLS = ("SeniorCitizen", "tenure")
FLOAT_COLS = ("MonthlyCharges", "TotalCharges")


def fila_a_json(path: str, fila: int) -> dict:
    """Convierte una fila del CSV en el JSON de /predict: números como números, texto como texto,
    celdas vacías (p. ej. TotalCharges de un cliente nuevo) como null."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"El archivo {path} no tiene filas.")
    if not 0 <= fila < len(rows):
        sys.exit(f"--fila debe estar entre 0 y {len(rows) - 1}.")
    registro = {}
    for k, v in rows[fila].items():
        v = v.strip()
        if v == "":
            registro[k] = None
        elif k in INT_COLS:
            registro[k] = int(v)
        elif k in FLOAT_COLS:
            registro[k] = float(v)
        else:
            registro[k] = v
    return registro


def llamar(metodo: str, url: str, cuerpo=None, timeout: int = 120):
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    req = urllib.request.Request(url, data=datos, method=metodo, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")
    except urllib.error.URLError as exc:
        sys.exit(f"No se pudo conectar con {url}: {exc.reason}. ¿Está la API en ejecución?")


def mostrar(titulo, status, cuerpo):
    print(f"--- {titulo}\nHTTP {status}\n{json.dumps(cuerpo, indent=2, ensure_ascii=False)}\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cliente de prueba de la API de churn.")
    p.add_argument("accion", choices=["health", "versions", "predict", "train", "demo"])
    p.add_argument("--url", default="http://localhost:8000", help="Base de la API (por defecto: %(default)s).")
    p.add_argument("--csv", default="data/ejemplo_prediccion.csv", help="CSV con registros de ejemplo.")
    p.add_argument("--fila", type=int, default=0, help="Fila del CSV a enviar (desde 0).")
    p.add_argument("--sin-campo", default=None, help="Quita esta variable del registro (prueba de validación).")
    p.add_argument("--version", default=None, help="Identificador para /train (por defecto, el siguiente libre).")
    a = p.parse_args(argv)
    base = a.url.rstrip("/")

    if a.accion == "health":
        mostrar("GET /health", *llamar("GET", f"{base}/health"))
    elif a.accion == "versions":
        mostrar("GET /versions", *llamar("GET", f"{base}/versions"))
    elif a.accion == "train":
        cuerpo = {"version": a.version} if a.version else None
        mostrar("POST /train", *llamar("POST", f"{base}/train", cuerpo))
    elif a.accion == "predict":
        reg = fila_a_json(a.csv, a.fila)
        if a.sin_campo:
            reg.pop(a.sin_campo, None)
        print("Solicitud enviada:\n" + json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
        mostrar("POST /predict", *llamar("POST", f"{base}/predict", reg))
    else:  # demo: dos válidas y una inválida
        for titulo, fila, quitar in (("Válida 1", 0, None), ("Válida 2", 1, None), ("Inválida (sin tenure)", 2, "tenure")):
            reg = fila_a_json(a.csv, fila)
            if quitar:
                reg.pop(quitar)
            mostrar(f"POST /predict - {titulo}", *llamar("POST", f"{base}/predict", reg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
