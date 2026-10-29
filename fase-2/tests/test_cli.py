import sys
import subprocess
import pandas as pd
import pytest

def test_cli_train_ejecucion_exitosa(tmp_path):
    """Prueba la ejecución de train.py desde la línea de comandos."""
    cmd = [
        sys.executable, "train.py",
        "--data", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "--target", "Churn",
        "--output-dir", str(tmp_path),
        "--version", "v_cli_test",
        "--seed", "42"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Error en train CLI: {result.stderr}"
    assert (tmp_path / "v_cli_test" / "modelo.joblib").exists() or (tmp_path / "v_cli_test" / "model.joblib").exists()

def test_cli_predict_ejecucion_exitosa(tmp_path):
    """Prueba la ejecución de predict.py desde la línea de comandos generando el CSV de salida."""
    # 1. Entrenar modelo
    train_cmd = [
        sys.executable, "train.py",
        "--data", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "--target", "Churn",
        "--output-dir", str(tmp_path),
        "--version", "v_predict_cli",
        "--seed", "42"
    ]
    subprocess.run(train_cmd, check=True)
    
    # Buscar el modelo generado (.joblib)
    v_dir = tmp_path / "v_predict_cli"
    model_file = list(v_dir.glob("*.joblib"))[0]
    
    output_csv = tmp_path / "predicciones.csv"
    predict_cmd = [
        sys.executable, "predict.py",
        "--model", str(model_file),
        "--input", "data/ejemplo_prediccion.csv",
        "--output", str(output_csv)
    ]
    
    result = subprocess.run(predict_cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Error en predict CLI: {result.stderr}"
    assert output_csv.exists()
    
    # Verificar cantidad de filas conservadas
    df_in = pd.read_csv("data/ejemplo_prediccion.csv")
    df_out = pd.read_csv(output_csv)
    assert len(df_in) == len(df_out)

def test_cli_entrada_invalida_codigo_salida_no_cero():
    """Verifica que una entrada inválida devuelva un código de salida distinto de 0."""
    predict_cmd = [
        sys.executable, "predict.py",
        "--model", "models/v_inexistente/modelo.joblib",
        "--input", "data/ejemplo_invalido.csv",
        "--output", "outputs/salida_fallida.csv"
    ]
    
    result = subprocess.run(predict_cmd, capture_output=True, text=True)
    assert result.returncode != 0