#!/usr/bin/env python
"""Entrena una versión nueva del modelo de churn y la guarda en models/<version>/.

Ejemplo (una sola línea):
    python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --output-dir models --version v1 --seed 42
"""
import argparse
import sys

from src.config import DEFAULT_DATA_SOURCE, ID_COLUMN, TARGET
from src.errors import ChurnError
from src.training import run_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entrena el modelo de churn (Regresión Logística) y guarda modelo, "
                    "preprocesamiento, métricas y metadatos en una versión nueva.",
        epilog="Una versión existente nunca se sobrescribe: use otro --version.")
    p.add_argument("--data", required=True, help="CSV con datos etiquetados.")
    p.add_argument("--target", default=TARGET, help="Variable objetivo (por defecto: %(default)s).")
    p.add_argument("--output-dir", default="models", help="Carpeta de versiones (por defecto: %(default)s).")
    p.add_argument("--version", required=True, help="Identificador de la versión (p. ej. v1).")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (por defecto: %(default)s).")
    p.add_argument("--test-size", type=float, default=0.2, help="Fracción para prueba (por defecto: %(default)s).")
    p.add_argument("--C", type=float, default=1.0, help="Inverso de la regularización L2 (por defecto: %(default)s).")
    p.add_argument("--max-iter", type=int, default=1000, help="Iteraciones máximas (por defecto: %(default)s).")
    p.add_argument("--id-column", default=ID_COLUMN, help="Identificador de cliente; se excluye del modelo.")
    p.add_argument("--optimize-threshold", action="store_true",
                   help="Elige el umbral que minimiza el costo con validación cruzada DENTRO de train.")
    p.add_argument("--cost-fn", type=float, default=5.0,
                   help="Costo relativo de un abandono no detectado (por defecto: %(default)s).")
    p.add_argument("--cost-fp", type=float, default=1.0,
                   help="Costo relativo de una falsa alarma / oferta de retención (por defecto: %(default)s).")
    p.add_argument("--cv-folds", type=int, default=5, help="Particiones para elegir el umbral (por defecto: %(default)s).")
    p.add_argument("--data-source", default=DEFAULT_DATA_SOURCE, help="Texto con la fuente de los datos (metadatos).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        res = run_training(
            data_path=args.data, target=args.target, output_dir=args.output_dir,
            version=args.version, seed=args.seed, test_size=args.test_size, C=args.C,
            max_iter=args.max_iter, optimize_threshold=args.optimize_threshold,
            cost_fn=args.cost_fn, cost_fp=args.cost_fp, cv_folds=args.cv_folds,
            id_column=args.id_column, data_source=args.data_source)
    except ChurnError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # error no previsto: mensaje claro y código distinto de cero
        print(f"ERROR inesperado ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    m, base = res["metrics"]["modelo"], res["metrics"]["baseline"]
    meta = res["metadata"]
    print(f"Versión '{meta['version']}' entrenada con {meta['particion']['n_entrenamiento']} clientes "
          f"y evaluada con {meta['particion']['n_prueba']} (umbral = {m['umbral']}).")
    print(f"{'':16s}{'accuracy':>10s}{'recall':>10s}{'f1':>10s}{'roc_auc':>10s}{'costo':>10s}")
    for name, r in (("Baseline", base), ("Modelo", m)):
        print(f"{name:16s}{r['accuracy']:>10.4f}{r['recall_churn']:>10.4f}{r['f1_churn']:>10.4f}"
              f"{r['roc_auc']:>10.4f}{r['costo_total']:>10.1f}")
    cm = m["matriz_confusion"]
    print(f"Abandonos detectados: {cm['tp']} de {cm['tp'] + cm['fn']} | no detectados: {cm['fn']} | falsas alarmas: {cm['fp']}")
    print("Archivos generados:")
    for k in ("modelo", "metricas", "metadatos"):
        print(f"  {res['paths'][k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
