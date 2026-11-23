#!/usr/bin/env bash
# Recorrido de verificación desde un directorio limpio (.verificacion/, se recrea en cada ejecución):
#   entrenar con un identificador nuevo -> cargar el artefacto y predecir -> revisar metadatos
#   (commit y entorno) -> comprobar que repetir el identificador NO reemplaza la versión.
# Uso (desde fase-2/):
#   bash verificar_recorrido.sh                 # recorrido local
#   bash verificar_recorrido.sh v-verif --docker  # además construye la imagen y repite el entrenamiento en Docker
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${1:-v-verificacion}"
DATA="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
WORK="$PWD/.verificacion"
rm -rf "$WORK" && mkdir -p "$WORK/models" "$WORK/outputs"
HEAD_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo no-disponible)"

sha() { python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
campo() { python -c "import json,sys;m=json.load(open(sys.argv[1],encoding='utf-8'));print(eval(sys.argv[2]))" "$1" "$2"; }

echo "== 1. Entrenar '$VERSION' en un directorio limpio"
python train.py --data "$DATA" --target Churn --output-dir "$WORK/models" --version "$VERSION" --seed 42

echo; echo "== 2. Archivos de la versión"
ls -l "$WORK/models/$VERSION"

echo; echo "== 3. Cargar el artefacto y predecir sobre un CSV pequeño"
python predict.py --model "$WORK/models/$VERSION/modelo.joblib" --input data/ejemplo_prediccion.csv \
  --output "$WORK/outputs/predicciones.csv"
head -4 "$WORK/outputs/predicciones.csv"

echo; echo "== 4. Metadatos: commit y entorno"
META="$WORK/models/$VERSION/metadata.json"
echo "commit en metadata.json : $(campo "$META" "m['codigo']['commit']")"
echo "commit actual (HEAD)    : $HEAD_COMMIT"
echo "python / scikit-learn   : $(campo "$META" "m['entorno']['python']") / $(campo "$META" "m['entorno']['scikit-learn']")"
echo "semilla / datos sha256  : $(campo "$META" "m['semilla']") / $(campo "$META" "m['datos']['sha256'][:12]")"

echo; echo "== 5. Repetir el identificador no reemplaza la versión"
ANTES="$(sha "$WORK/models/$VERSION/modelo.joblib")"
if python train.py --data "$DATA" --output-dir "$WORK/models" --version "$VERSION" --seed 7 2>"$WORK/error.txt"; then
  echo "FALLO: se aceptó un identificador repetido"; exit 1
else
  echo "Rechazado con código de salida $? -> $(cat "$WORK/error.txt")"
fi
DESPUES="$(sha "$WORK/models/$VERSION/modelo.joblib")"
[ "$ANTES" = "$DESPUES" ] && echo "modelo.joblib intacto (sha256 ${ANTES:0:12})" || { echo "FALLO: el modelo cambió"; exit 1; }

if [ "${2:-}" = "--docker" ]; then
  echo; echo "== 6. Docker: el commit llega a los metadatos"
  docker build -t modelos-fase2:1.0 --build-arg GIT_COMMIT="$HEAD_COMMIT" .
  mkdir -p "$WORK/models-docker"
  MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD/data:/app/data:ro" -v "$WORK/models-docker:/app/models" \
    modelos-fase2:1.0 python train.py --data "$DATA" --target Churn --output-dir models --version "$VERSION-docker" --seed 42
  DC="$(campo "$WORK/models-docker/$VERSION-docker/metadata.json" "m['codigo']['commit']")"
  echo "commit en metadata.json (Docker): $DC"
  [ "$DC" = "$HEAD_COMMIT" ] && echo "OK: coincide con HEAD" || { echo "FALLO: no coincide con HEAD"; exit 1; }
fi
echo; echo "Recorrido completado. Evidencia en: $WORK"
