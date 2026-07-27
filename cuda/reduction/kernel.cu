/**
 * @file kernel.cu
 * @brief Implementación del kernel CUDA de reducción (suma de un vector).
 *
 * POR QUÉ LA REDUCCIÓN NO ES TRIVIAL EN GPU:
 * En CPU es un bucle secuencial (cada suma depende de la anterior). En GPU
 * miles de hilos no pueden acumular sobre una misma variable sin pisarse.
 * La solución clásica es la REDUCCIÓN EN ÁRBOL: en cada paso, la mitad de
 * los hilos suma su valor con el de otro hilo, y el número de valores se
 * reduce a la mitad → log2(n) pasos en lugar de n.
 *
 * ESTRATEGIA DE DOS ETAPAS (dos lanzamientos del MISMO kernel):
 *
 *   Etapa 1:  256 bloques × 256 hilos sobre los N elementos
 *             → 256 sumas parciales (una por bloque)
 *   Etapa 2:  1 bloque × 256 hilos sobre esas 256 parciales
 *             → 1 float: la suma total
 *
 * ¿Por qué dos etapas? __syncthreads() solo sincroniza DENTRO de un bloque:
 * no existe barrera global entre bloques dentro de un kernel. La única
 * sincronización global garantizada es el FIN de un kernel, así que la
 * combinación de parciales de distintos bloques exige un segundo
 * lanzamiento. Este límite arquitectónico (idéntico en OpenCL) es
 * precisamente lo que este benchmark aporta al estudio: mide el coste de la
 * cooperación y sincronización entre hilos, que en Vector Addition no existe.
 *
 * GEOMETRÍA FIJA (256×256) + BUCLE GRID-STRIDE:
 * En lugar de lanzar ceil(N/256) bloques (dependiente de N), se lanzan
 * SIEMPRE 256 bloques y cada hilo acumula secuencialmente los elementos
 * i, i+65536, i+131072, ... ("grid-stride loop"). Ventajas:
 *   1. El buffer de parciales tiene tamaño fijo (256 floats) para todo N.
 *   2. La geometría es idéntica en los 5 tamaños del barrido: el tiempo
 *      medido escala solo con N, no con la configuración de lanzamiento.
 *   3. Permite reutilizar el mismo kernel para la etapa 2 (con 1 bloque).
 */

#include "kernel.cuh"

#include "cuda/common/cuda_utils.cuh"

namespace bench {
namespace cuda {

namespace {

/// Hilos por bloque: mismo valor que en los demás benchmarks del proyecto
/// (múltiplo del warp de 32; ver justificación en vector_add/kernel.cu).
/// Debe ser potencia de 2: la reducción en árbol divide por 2 en cada paso.
constexpr unsigned kThreadsPerBlock = 256u;

/**
 * @brief Kernel: reduce `elementCount` floats a UNA suma parcial por bloque.
 *
 * Tres fases dentro de cada bloque:
 *
 *   1. ACUMULACIÓN SECUENCIAL (grid-stride): cada hilo suma "sus" elementos
 *      del vector en un registro privado. Aquí no hace falta sincronizar:
 *      cada hilo trabaja solo, con accesos coalescidos (hilos consecutivos
 *      leen posiciones consecutivas en cada vuelta del bucle).
 *
 *   2. VOLCADO A MEMORIA COMPARTIDA + ÁRBOL: los 256 acumuladores privados
 *      pasan a __shared__ y se combinan en árbol binario: 128 hilos activos,
 *      luego 64, 32, ... 1 (8 pasos = log2(256)). La barrera __syncthreads()
 *      entre pasos es OBLIGATORIA: garantiza que shared[tid + s] ya fue
 *      escrito antes de leerlo. El patrón "secuencial" (tid < s, hilos
 *      activos contiguos) evita divergencia dentro de los warps y conflictos
 *      de bancos de memoria compartida.
 *
 *   3. ESCRITURA: el hilo 0 de cada bloque escribe la parcial de su bloque.
 *
 * El MISMO kernel sirve para la etapa 2: con gridDim.x == 1 y la entrada
 * apuntando a las 256 parciales, el resultado es la suma total.
 */
__global__ void reduceSumKernel(const float* __restrict__ input,
                                float* __restrict__ partials,
                                std::size_t elementCount) {
    __shared__ float shared[kThreadsPerBlock];

    const unsigned tid = threadIdx.x;

    // --- Fase 1: acumulación secuencial con paso de grid -----------------
    const std::size_t gridStride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    float acc = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
         i < elementCount; i += gridStride) {
        acc += input[i];
    }

    // --- Fase 2: reducción en árbol dentro del bloque --------------------
    shared[tid] = acc;
    __syncthreads();  // Todos los acumuladores deben estar volcados.

    for (unsigned s = kThreadsPerBlock / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();  // El paso siguiente lee lo que este escribió.
    }

    // --- Fase 3: una suma parcial por bloque -----------------------------
    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}

}  // namespace

void launchReduction(const float* deviceInput,
                     float*       devicePartials,
                     float*       deviceResult,
                     std::size_t  elementCount,
                     cudaStream_t stream) {
    // Etapa 1: N elementos → kReductionStage1Blocks sumas parciales.
    reduceSumKernel<<<kReductionStage1Blocks, kThreadsPerBlock, 0, stream>>>(
        deviceInput, devicePartials, elementCount);
    CUDA_CHECK(cudaGetLastError());

    // Etapa 2: parciales → suma total. Un solo bloque: la barrera interna
    // basta para combinar las 256 parciales. Ambos lanzamientos van al MISMO
    // stream, que garantiza el orden (la etapa 2 no arranca hasta terminar
    // la 1) sin sincronizar con la CPU.
    reduceSumKernel<<<1, kThreadsPerBlock, 0, stream>>>(
        devicePartials, deviceResult, kReductionStage1Blocks);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace bench
