"""Fase 4 - Script de monitoreo: total de solicitudes y porcentaje de fallos."""
import subprocess
import sys
from pathlib import Path

from src.monitoring import compute_metrics, format_report, read_log

ROOT = Path(__file__).resolve().parent.parent
HEADER = "fecha_hora,estado_http,version,prediccion\n"


def escribir(path, filas):
    path.write_text(HEADER + "".join(filas), encoding="utf-8")


def test_dos_validas_y_una_fallida_dan_3_solicitudes_y_33_3_por_ciento(tmp_path):
    log = tmp_path / "log.csv"
    escribir(log, ["2026-11-06T10:00:00,200,v1,Churn\n", "2026-11-06T10:01:00,200,v1,No Churn\n",
                   "2026-11-06T10:02:00,422,,\n"])
    m = compute_metrics(read_log(log))
    assert (m["total"], m["fallidas"], m["porcentaje_fallos"]) == (3, 1, 33.3)
    texto = format_report(m, log)
    assert "Total de solicitudes: 3" in texto and "Porcentaje de fallos: 33,3 %" in texto


def test_registro_vacio_o_inexistente_muestra_total_0_y_sin_solicitudes(tmp_path):
    vacio = tmp_path / "vacio.csv"
    escribir(vacio, [])                                       # solo encabezado
    for p in (vacio, tmp_path / "no_existe.csv"):
        m = compute_metrics(read_log(p))
        assert m["total"] == 0 and m["porcentaje_fallos"] is None
        t = format_report(m, p)
        assert "Total de solicitudes: 0" in t and "sin solicitudes" in t


def test_cuenta_como_fallidos_solo_4xx_y_5xx(tmp_path):
    log = tmp_path / "log.csv"
    escribir(log, [f"2026-11-06T10:00:0{i},{c},,\n" for i, c in enumerate((200, 201, 400, 422, 500, 503, 399))])
    m = compute_metrics(read_log(log))
    assert m["total"] == 7 and m["fallidas"] == 4


def test_estado_no_numerico_se_avisa_y_no_cuenta_como_fallo(tmp_path):
    log = tmp_path / "log.csv"
    escribir(log, ["2026-11-06T10:00:00,200,v1,Churn\n", "2026-11-06T10:01:00,abc,,\n"])
    m = compute_metrics(read_log(log))
    assert m["total"] == 2 and m["fallidas"] == 0 and m["filas_con_estado_invalido"] == 1
    assert "AVISO" in format_report(m, log)


def test_script_por_linea_de_comandos(tmp_path):
    log = tmp_path / "log.csv"
    escribir(log, ["2026-11-06T10:00:00,200,v1,Churn\n", "2026-11-06T10:01:00,200,v1,No Churn\n",
                   "2026-11-06T10:02:00,422,,\n"])
    r = subprocess.run([sys.executable, "monitor.py", "--log", str(log)], cwd=ROOT, text=True, capture_output=True)
    assert r.returncode == 0
    assert "Total de solicitudes: 3" in r.stdout and "33,3 %" in r.stdout
    r = subprocess.run([sys.executable, "monitor.py", "--log", str(tmp_path / "nada.csv")],
                       cwd=ROOT, text=True, capture_output=True)
    assert r.returncode == 0 and "sin solicitudes" in r.stdout
