/**
 * @file kernel.cu
 * @brief Implementación del kernel CUDA de suma de vectores.
 *
 * CONCEPTOS CUDA QUE APARECEN AQUÍ (guía de aprendizaje):
 *
 *   __global__   Marca una función como "kernel": código que se ejecuta en
 *                la GPU pero se INVOCA desde la CPU. Miles de hilos ejecutan
 *                esta misma función en paralelo, cada uno sobre datos
 *                distintos (modelo SIMT: Single Instruction, Multiple Threads).
 *
 *   Jerarquía de ejecución:
 *                grid  →  bloques  →  hilos
 *                Un lanzamiento crea un GRID de BLOQUES; cada bloque contiene
 *                el mismo número de HILOS. Los bloques se reparten entre los
 *                multiprocesadores (SMs) de la GPU (la Quadro K2200 tiene 5).
 *
 *   Variables implícitas dentro del kernel:
 *                threadIdx.x  índice del hilo dentro de su bloque
 *                blockIdx.x   índice del bloque dentro del grid
 *                blockDim.x   número de hilos por bloque
 *                Con ellas cada hilo calcula QUÉ elemento le toca procesar.
 */

#include "kernel.cuh"

#include "cuda/common/cuda_utils.cuh"

namespace bench {
namespace cuda {

namespace {

/**
 * @brief Hilos por bloque para este kernel.
 *
 * JUSTIFICACIÓN (Maxwell / Quadro K2200):
 *   - Debe ser múltiplo de 32 (el "warp": la unidad mínima de ejecución;
 *     los hilos se ejecutan en grupos de 32 al unísono).
 *   - 256 es el valor de consenso para kernels limitados por ancho de banda
 *     de memoria como éste: suficientes warps por SM para ocultar la
 *     latencia de memoria, sin agotar registros ni limitar la ocupación.
 *   - Está definido en UN solo lugar: cambiarlo para experimentar no toca
 *     ninguna otra parte del código.
 */
constexpr unsigned kThreadsPerBlock = 256u;

/**
 * @brief Kernel: cada hilo calcula UN elemento de C.
 *
 * Mapeo "un hilo por elemento", el más simple y transparente para un
 * benchmark de referencia:
 *
 *     índice global = blockIdx.x * blockDim.x + threadIdx.x
 *
 * Ejemplo con blockDim.x = 4 (didáctico):
 *     bloque 0: hilos 0,1,2,3   → elementos 0..3
 *     bloque 1: hilos 0,1,2,3   → elementos 4..7   (1*4 + 0..3)
 *
 * LA GUARDA (i < elementCount) ES OBLIGATORIA:
 * El grid se redondea hacia arriba, así que casi siempre hay más hilos que
 * elementos. Los hilos sobrantes no deben tocar memoria: sin esta guarda
 * escribirían fuera del buffer (comportamiento indefinido / corrupción).
 *
 * __restrict__ promete al compilador que los punteros no se solapan entre
 * sí, lo que le permite optimizar las cargas de memoria. Es cierto aquí
 * porque A, B y C son reservas independientes.
 */
__global__ void vectorAddKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                std::size_t elementCount) {
    // Índice global de este hilo dentro de todo el grid.
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Guarda de límites: solo los hilos con índice válido trabajan.
    if (i < elementCount) {
        c[i] = a[i] + b[i];  // El algoritmo completo: una suma por hilo.
    }
}

}  // namespace (anónimo: kernel y constante son detalles privados del módulo)

void launchVectorAdd(const float* deviceA,
                     const float* deviceB,
                     float*       deviceC,
                     std::size_t  elementCount,
                     cudaStream_t stream) {
    // Geometría del grid: bloques suficientes para cubrir TODOS los elementos.
    // La división redondea hacia arriba (patrón ceil-div): para N = 1000 y
    // 256 hilos/bloque → (1000 + 255) / 256 = 4 bloques = 1024 hilos
    // (los 24 sobrantes los descarta la guarda del kernel).
    const unsigned blocksPerGrid = static_cast<unsigned>(
        (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock);

    // Sintaxis de lanzamiento CUDA: kernel<<<bloques, hilos, memCompartida, stream>>>
    // Es ASÍNCRONO: la CPU continúa inmediatamente; la medición del tiempo
    // corre a cargo de los eventos CUDA que rodean esta llamada en main.cu.
    vectorAddKernel<<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(
        deviceA, deviceB, deviceC, elementCount);

    // Un lanzamiento inválido (geometría imposible, punteros nulos...) NO
    // lanza excepción: deja un código de error pendiente. Lo recogemos aquí
    // para fallar rápido y con diagnóstico, en vez de arrastrar el error.
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace bench
