/**
 * @file kernel.cu
 * @brief Implementación del kernel CUDA de multiplicación de matrices.
 *
 * CONCEPTOS CUDA NUEVOS RESPECTO DE vector_add (guía de aprendizaje):
 *
 *   __shared__   Memoria COMPARTIDA: una memoria en el propio chip del SM
 *                (latencia ~100× menor que la GDDR5 global), visible por
 *                todos los hilos de un MISMO bloque. Es la herramienta clave
 *                de este kernel: permite reutilizar datos entre hilos.
 *
 *   __syncthreads()
 *                Barrera de sincronización DENTRO de un bloque: ningún hilo
 *                continúa hasta que todos la alcanzan. Obligatoria cada vez
 *                que unos hilos leen lo que otros escribieron en __shared__.
 *
 *   Grid 2D      El problema es bidimensional (cada elemento de C tiene fila
 *                y columna), así que el grid y los bloques son 2D: dim3.
 *
 * POR QUÉ TILING (y no el kernel "naive"):
 * El elemento C[fila][col] necesita la fila completa de A y la columna
 * completa de B (2N lecturas). Sin cooperación, el kernel completo haría
 * 2N³ lecturas de memoria global. Con TILING, cada bloque de 16×16 hilos
 * carga por turnos "baldosas" (tiles) de 16×16 de A y B en memoria
 * compartida; cada dato cargado lo REUTILIZAN los 16 hilos de su fila o
 * columna, dividiendo el tráfico a memoria global por 16. Esto convierte el
 * benchmark en un test de cómputo con cooperación entre hilos — el patrón
 * que lo diferencia de Vector Addition (puro ancho de banda) dentro del
 * estudio comparativo.
 */

#include "kernel.cuh"

#include "cuda/common/cuda_utils.cuh"

namespace bench {
namespace cuda {

namespace {

/**
 * @brief Lado de la baldosa (tile) y del bloque de hilos: 16×16 = 256 hilos.
 *
 * JUSTIFICACIÓN:
 *   - 256 hilos por bloque: el MISMO total que en vector_add (y que en la
 *     versión OpenCL), para mantener la geometría comparable entre
 *     benchmarks y frameworks.
 *   - 16 es múltiplo de la anchura de acceso coalescido y hace que cada
 *     baldosa (16×16 floats = 1 KB) quepa holgadamente en los 64 KB de
 *     memoria compartida por SM de Maxwell (dos baldosas por bloque = 2 KB).
 */
constexpr unsigned kTileSize = 16u;

/**
 * @brief Kernel: cada hilo calcula UN elemento C[fila][col].
 *
 * Estructura en fases (bucle sobre baldosas t = 0 .. N/16):
 *   1. CARGA cooperativa: cada hilo trae UN elemento de la baldosa t de A y
 *      UN elemento de la baldosa t de B a memoria compartida.
 *   2. __syncthreads(): nadie computa hasta que la baldosa esté completa.
 *   3. CÓMPUTO: cada hilo acumula el producto parcial de 16 términos usando
 *      SOLO memoria compartida.
 *   4. __syncthreads(): nadie sobreescribe la baldosa hasta que todos
 *      terminaron de leerla.
 *
 * LAS GUARDAS (fila < n, columna < n) siguen la misma lógica que en
 * vector_add: si N no es múltiplo de 16, los hilos fuera de rango cargan 0
 * (neutro de la suma) y no escriben resultado. Cargar 0 en vez de saltarse
 * la carga mantiene la baldosa completa y las barreras uniformes (TODOS los
 * hilos deben alcanzar __syncthreads(), sin divergencia condicional).
 *
 * ORDEN DE ACUMULACIÓN: k recorre 0..N-1 en orden ascendente (baldosa a
 * baldosa), el MISMO orden que la referencia CPU. Aun así el resultado no es
 * idéntico bit a bit: nvcc contrae a*b+c en instrucciones FMA (un solo
 * redondeo) y la CPU usa multiplicación y suma separadas (dos redondeos).
 * De ahí la tolerancia relativa de la verificación (ver main.cu).
 */
__global__ void matrixMulKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                unsigned n) {
    // Baldosas en memoria compartida: se rellenan cooperativamente en cada
    // iteración del bucle y las leen los 256 hilos del bloque.
    __shared__ float tileA[kTileSize][kTileSize];
    __shared__ float tileB[kTileSize][kTileSize];

    // Coordenadas del elemento de C asignado a este hilo.
    // Convención CUDA: .x recorre columnas (memoria contigua → coalescido).
    const unsigned row = blockIdx.y * kTileSize + threadIdx.y;
    const unsigned col = blockIdx.x * kTileSize + threadIdx.x;

    float acc = 0.0f;  // Acumulador del producto escalar (en registro).

    const unsigned numTiles = (n + kTileSize - 1) / kTileSize;
    for (unsigned t = 0; t < numTiles; ++t) {
        // --- Fase 1: carga cooperativa de las baldosas t de A y B --------
        const unsigned aCol = t * kTileSize + threadIdx.x;
        const unsigned bRow = t * kTileSize + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < n && aCol < n) ? a[static_cast<std::size_t>(row) * n + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < n && col < n) ? b[static_cast<std::size_t>(bRow) * n + col] : 0.0f;

        // --- Fase 2: barrera — la baldosa debe estar completa ------------
        __syncthreads();

        // --- Fase 3: producto parcial usando solo memoria compartida -----
        for (unsigned k = 0; k < kTileSize; ++k) {
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // --- Fase 4: barrera — nadie recarga hasta que todos leyeron -----
        __syncthreads();
    }

    // Solo los hilos con coordenadas válidas escriben su elemento.
    if (row < n && col < n) {
        c[static_cast<std::size_t>(row) * n + col] = acc;
    }
}

}  // namespace

void launchMatrixMul(const float* deviceA,
                     const float* deviceB,
                     float*       deviceC,
                     std::size_t  dimension,
                     cudaStream_t stream) {
    const unsigned n = static_cast<unsigned>(dimension);

    // Geometría 2D: un bloque de 16×16 hilos por cada baldosa de C.
    // Mismo patrón ceil-div que en vector_add, ahora en dos dimensiones.
    const unsigned blocksPerSide = (n + kTileSize - 1) / kTileSize;
    const dim3 blockDim(kTileSize, kTileSize);
    const dim3 gridDim(blocksPerSide, blocksPerSide);

    matrixMulKernel<<<gridDim, blockDim, 0, stream>>>(deviceA, deviceB, deviceC, n);

    // Recoger un posible error de lanzamiento (misma política fail-fast).
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace bench
