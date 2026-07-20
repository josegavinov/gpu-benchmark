# GPU Benchmark Suite — CUDA vs OpenCL

Infraestructura experimental reutilizable para la comparación de rendimiento
entre **CUDA** y **OpenCL** ejecutando los mismos algoritmos bajo idénticas
condiciones, con calidad de artículo IEEE.

## Entorno experimental

| Componente | Detalle |
|---|---|
| GPU | NVIDIA Quadro K2200 (Maxwell, 4 GB GDDR5) |
| CPU | Intel Xeon E5-2699 v3 (72 hilos lógicos) |
| RAM | 31 GB |
| SO | Ubuntu 22.04.5 LTS |
| CUDA Toolkit | 11.8 |
| Driver NVIDIA | 535.309.01 |

## Estructura del proyecto

```
gpu-benchmark/
├── include/common/        # Infraestructura agnóstica al framework (C++ puro, header-only)
│   ├── benchmark_config.hpp   # Parseo y validación de línea de comandos
│   ├── csv_logger.hpp         # Registro automático de resultados en CSV
│   ├── data_init.hpp          # Generación determinista de datos (semilla fija)
│   └── verification.hpp       # Verificación CPU vs GPU (PASS/FAIL)
├── cuda/
│   ├── common/                # Utilidades específicas de CUDA
│   │   └── cuda_utils.cuh     # CUDA_CHECK, EventTimer (CUDA Events), info GPU
│   ├── vector_add/            # Benchmark 1: C[i] = A[i] + B[i]
│   ├── matrix_mul/            # (futuro)
│   └── reduction/             # (futuro)
├── opencl/                    # Fase 2: espejo de cuda/ usando OpenCL
├── scripts/                   # Automatización Bash (barridos de tamaño × repeticiones)
├── results/
│   ├── cuda/                  # CSVs generados por benchmarks CUDA
│   └── opencl/                # CSVs generados por benchmarks OpenCL
├── docs/                      # Protocolo experimental, decisiones de diseño
└── README.md
```

### Principio de diseño central

Todo lo que **no** depende del framework (argumentos, datos, verificación,
CSV) vive en `include/common/` y se comparte entre CUDA y OpenCL. Así, la
única diferencia entre ambas implementaciones es el código GPU en sí —
exactamente la variable que la investigación quiere aislar.

## Compilación (servidor)

En el servidor, CMake no localiza el toolkit CUDA automáticamente: hay que
indicar el compilador de forma explícita al configurar (solo la primera vez;
queda guardado en la caché de CMake):

```bash
cd gpu-benchmark/cuda/vector_add
mkdir -p build && cd build
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
make
```

## Uso

Cada benchmark realiza **una** medición por ejecución; las repeticiones las
orquesta un script externo:

```bash
./vector_add <tamano_problema> [iteracion] [ruta_csv]

# Ejemplos
./vector_add 100000
./vector_add 5000000 12 results/cuda/vector_add.csv
```

Salida: `PASS`/`FAIL` por stdout y una fila añadida al CSV con el esquema:

```
Framework,Benchmark,ProblemSize,Iteration,HostToDevice_ms,Kernel_ms,DeviceToHost_ms,Total_ms,Status
```

## Barrido experimental completo

`scripts/run_benchmark.sh` ejecuta el protocolo completo (5 tamaños × 30
repeticiones) invocando el binario una vez por medición. Es genérico: sirve
para cualquier benchmark y framework. Se puede invocar desde cualquier
carpeta (se ancla solo a la raíz del repo). Si el CSV ya existe, lo archiva
con timestamp antes de empezar (nunca mezcla corridas).

```bash
chmod +x scripts/run_benchmark.sh   # solo la primera vez

./scripts/run_benchmark.sh cuda/vector_add/build/vector_add \
                           results/cuda/vector_add.csv

# Barrido reducido de prueba (tamaños y repeticiones a medida):
SIZES="1000 2000" REPS=3 ./scripts/run_benchmark.sh \
    cuda/vector_add/build/vector_add results/cuda/vector_add.csv
```

## Estado del proyecto

- [x] Fase 1a — Infraestructura común (CSV, datos, verificación, timing)
- [x] Fase 1b — Vector Addition en CUDA (validado en el servidor: 2026-07-19)
- [x] Fase 1c — Scripts de automatización (validado: 150/150 PASS en 33 s)
- [x] Fase 2 — Vector Addition en OpenCL (validado en el servidor: 2026-07-19)
- [ ] Fase 3 — Matrix Multiplication, Reduction, Prefix Sum
