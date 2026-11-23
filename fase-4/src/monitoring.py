"""Métricas de operación calculadas a partir del registro de /predict (Fase 4).

Describen el FUNCIONAMIENTO del servicio, no la calidad predictiva (para esa se
necesitarían los resultados reales de los clientes).
"""
import csv
from collections import Counter
from pathlib import Path


def read_log(path) -> list:
    """Filas de datos del registro. Si el archivo no existe o está vacío -> lista vacía."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def compute_metrics(rows: list) -> dict:
    """Total de solicitudes y porcentaje de fallidas (HTTP 4xx o 5xx)."""
    total, failed, invalid = len(rows), 0, 0
    by_version = Counter()
    for r in rows:
        try:
            status = int(str(r.get("estado_http", "")).strip())
        except ValueError:
            invalid += 1
            continue
        if 400 <= status <= 599:
            failed += 1
        elif r.get("version"):
            by_version[r["version"]] += 1
    pct = None if total == 0 else round(failed / total * 100, 1)
    return {"total": total, "fallidas": failed, "porcentaje_fallos": pct,
            "filas_con_estado_invalido": invalid, "exitosas_por_version": dict(by_version)}


def format_report(m: dict, path) -> str:
    pct = "sin solicitudes" if m["porcentaje_fallos"] is None else f"{m['porcentaje_fallos']:.1f} %".replace(".", ",")
    lines = [f"Registro: {path}",
             f"Total de solicitudes: {m['total']}",
             f"Solicitudes fallidas (HTTP 4xx/5xx): {m['fallidas']}",
             f"Porcentaje de fallos: {pct}"]
    if m["exitosas_por_version"]:
        lines.append("Respuestas exitosas por versión: " +
                     ", ".join(f"{v}: {n}" for v, n in sorted(m["exitosas_por_version"].items())))
    if m["filas_con_estado_invalido"]:
        lines.append(f"AVISO: {m['filas_con_estado_invalido']} fila(s) con estado_http no numérico (no cuentan como fallidas).")
    return "\n".join(lines)
