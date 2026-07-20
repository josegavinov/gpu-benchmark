#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — Orquestador de barridos experimentales.
#
# Ejecuta un binario de benchmark (CUDA u OpenCL) sobre una lista de tamaños
# de problema, repitiendo cada tamaño un número fijo de veces. Cada ejecución
# es un PROCESO INDEPENDIENTE (decisión metodológica: sin estado residual
# entre repeticiones — ver benchmark_config.hpp).
#
# USO:
#   ./scripts/run_benchmark.sh <ruta_binario> <ruta_csv>
#
# EJEMPLO (Vector Addition CUDA):
#   ./scripts/run_benchmark.sh cuda/vector_add/build/vector_add \
#                              results/cuda/vector_add.csv
#
# Las rutas relativas se interpretan desde la RAÍZ del repositorio
# (el script se ancla solo, da igual desde dónde lo invoques).
#
# CONFIGURACIÓN DEL BARRIDO (sobreescribible por variables de entorno):
#   SIZES="1000 2000" REPS=5 ./scripts/run_benchmark.sh <binario> <csv>
# =============================================================================

set -u  # Variable no definida = error (típico bug silencioso de Bash).
# NOTA: no usamos "set -e" a propósito — un FAIL del benchmark (exit != 0)
# no debe abortar el barrido: se cuenta, se registra y se continúa.

# -----------------------------------------------------------------------------
# Protocolo experimental: tamaños y repeticiones.
# 5 tamaños × 30 repeticiones = 150 mediciones por benchmark y framework.
# 30 repeticiones permiten estadística robusta (media, desviación, IC 95%).
# -----------------------------------------------------------------------------
SIZES="${SIZES:-100000 500000 1000000 5000000 10000000}"
REPS="${REPS:-30}"

# -----------------------------------------------------------------------------
# Anclaje a la raíz del repositorio.
# El script vive en <raíz>/scripts/, así que la raíz es su carpeta padre.
# Todo lo demás (binario, CSV) se resuelve desde ahí: reproducible sin
# importar el directorio de trabajo del invocador.
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

# -----------------------------------------------------------------------------
# Argumentos.
# -----------------------------------------------------------------------------
if [[ $# -ne 2 ]]; then
    echo "Uso: $0 <ruta_binario> <ruta_csv>" >&2
    echo "Ej.: $0 cuda/vector_add/build/vector_add results/cuda/vector_add.csv" >&2
    exit 1
fi

BINARY="$1"
CSV="$2"

if [[ ! -x "${BINARY}" ]]; then
    echo "[ERROR] No existe o no es ejecutable: ${BINARY}" >&2
    echo "        ¿Compilaste? (cmake + make en la carpeta build del benchmark)" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Respaldo del CSV previo.
# El barrido asigna iteraciones 1..REPS por tamaño. Anexar sobre una corrida
# anterior duplicaría claves (tamaño, iteración) y contaminaría el análisis.
# Política: NUNCA borrar datos experimentales — se archivan con timestamp.
# -----------------------------------------------------------------------------
if [[ -f "${CSV}" ]]; then
    BACKUP="${CSV}.bak-$(date +%Y%m%d-%H%M%S)"
    mv "${CSV}" "${BACKUP}"
    echo "[INFO] CSV previo archivado como: ${BACKUP}"
fi

# -----------------------------------------------------------------------------
# Trazabilidad: dejar constancia de las condiciones de la corrida.
# -----------------------------------------------------------------------------
echo "============================================================"
echo " Barrido experimental"
echo "   Binario   : ${BINARY}"
echo "   CSV       : ${CSV}"
echo "   Tamanos   : ${SIZES}"
echo "   Reps/tam  : ${REPS}"
echo "   Host      : $(hostname)"
echo "   Fecha     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# -----------------------------------------------------------------------------
# Barrido: para cada tamaño, REPS ejecuciones independientes.
# stdout/stderr del binario se silencian (todo lo que importa va al CSV);
# solo mostramos progreso compacto y los fallos.
# -----------------------------------------------------------------------------
TOTAL_RUNS=0
FAILED_RUNS=0
START_SECONDS=${SECONDS}

for size in ${SIZES}; do
    echo -n "[RUN ] N=${size}: "
    for (( iter=1; iter<=REPS; iter++ )); do
        if "./${BINARY}" "${size}" "${iter}" "${CSV}" > /dev/null 2>&1; then
            echo -n "."          # Una marca por repetición correcta.
        else
            echo -n "F"          # F = FAIL (ya quedó registrado en el CSV).
            FAILED_RUNS=$(( FAILED_RUNS + 1 ))
        fi
        TOTAL_RUNS=$(( TOTAL_RUNS + 1 ))
    done
    echo "  (${REPS}/${REPS})"
done

ELAPSED=$(( SECONDS - START_SECONDS ))

# -----------------------------------------------------------------------------
# Resumen final y código de salida.
# Filas esperadas en el CSV = cabecera + TOTAL_RUNS.
# -----------------------------------------------------------------------------
CSV_ROWS=$(( $(wc -l < "${CSV}") - 1 ))

echo "============================================================"
echo " Resumen"
echo "   Ejecuciones : ${TOTAL_RUNS} (fallidas: ${FAILED_RUNS})"
echo "   Filas CSV   : ${CSV_ROWS} (esperadas: ${TOTAL_RUNS})"
echo "   Duracion    : ${ELAPSED} s"
echo "============================================================"

if [[ ${FAILED_RUNS} -gt 0 ]]; then
    echo "[WARN] Hubo ${FAILED_RUNS} ejecuciones con FAIL — revisar el CSV." >&2
    exit 1
fi
if [[ ${CSV_ROWS} -ne ${TOTAL_RUNS} ]]; then
    echo "[WARN] El numero de filas del CSV no coincide con las ejecuciones." >&2
    exit 1
fi

echo "[OK] Barrido completo sin fallos."
