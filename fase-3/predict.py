#!/usr/bin/env python
"""Genera predicciones con una versión guardada del modelo (no reentrena).

Ejemplo (una sola línea):
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""
import argparse
import sys
import warnings

from src.config import ID_COLUMN
from src.errors import ChurnError
from src.inference import run_prediction


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predice abandono (churn) para clientes nuevos con una versión guardada.",
        epilog="Salida: prediccion (1 = abandona/Churn, 0 = permanece/No Churn), "
               "prediccion_etiqueta y probabilidad_churn, en el mismo orden de las filas de entrada.")
    p.add_argument("--model", required=True, help="Ruta a modelo.joblib o a la carpeta de la versión (p. ej. models/v1).")
    p.add_argument("--input", required=True, help="CSV con las variables de entrada (sin variable objetivo).")
    p.add_argument("--output", required=True, help="CSV de salida con las predicciones.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Umbral de decisión (por defecto, el guardado en la versión).")
    p.add_argument("--id-column", default=ID_COLUMN, help="Identificador que se copia a la salida si existe.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = run_prediction(args.model, args.input, args.output, args.threshold, args.id_column)
    except ChurnError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR inesperado ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    for w in caught:
        print(f"AVISO: {w.message}", file=sys.stderr)
    rep = res["reporte"]
    if rep["columnas_extra_ignoradas"]:
        print(f"AVISO: columnas adicionales ignoradas: {rep['columnas_extra_ignoradas']}", file=sys.stderr)
    if rep["categorias_no_vistas"]:
        print(f"AVISO: categorías no vistas en entrenamiento (codificadas en ceros): {rep['categorias_no_vistas']}", file=sys.stderr)
    if rep["nulos_en_categoricas"]:
        print(f"AVISO: nulos en categóricas (tratados como categoría desconocida): {rep['nulos_en_categoricas']}", file=sys.stderr)
    if rep["totalcharges_imputados"]:
        print(f"AVISO: {rep['totalcharges_imputados']} valores de TotalCharges vacíos imputados con 0 "
              "(regla de la versión guardada).", file=sys.stderr)
    print(f"Versión '{res['version']}': {res['n_filas']} predicciones "
          f"({res['n_churn']} con riesgo de abandono, umbral {res['umbral']}).")
    print(f"Resultados en: {res['salida']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
