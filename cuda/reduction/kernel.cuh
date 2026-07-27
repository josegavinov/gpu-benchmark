/**
 * @file kernel.cuh
 * @brief Interfaz pública del kernel de reducción (suma de un vector) — CUDA.
 *
 * MISMA DECISIÓN DE DISEÑO QUE EN vector_add — OCULTAR EL LANZAMIENTO:
 * main.cu no ve la sintaxis <<<...>>> ni sabe que la reducción son DOS
 * lanzamientos encadenados (ver kernel.cu). Este wrapper encapsula la
 * estrategia completa; si mañana cambia (p. ej. a una sola pasada con
 * atomics), main.cu no se toca.
 *
 * CONVENCIÓN DEL BENCHMARK: reduce un vector de N floats a UN escalar
 * (la suma total). ProblemSize = N, mismos tamaños de barrido que
 * vector_add (100000 .. 10000000).
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace bench {
namespace cuda {

/// Número de bloques de la primera etapa — main.cu lo necesita para
/// dimensionar el buffer de sumas parciales (un float por bloque).
inline constexpr unsigned kReductionStage1Blocks = 256u;

/**
 * @brief Encola la reducción completa: suma de N floats → 1 float.
 *
 * Internamente son DOS kernels en el mismo stream (se ejecutan en orden):
 *   etapa 1: N elementos → kReductionStage1Blocks sumas parciales
 *   etapa 2: esas parciales → deviceResult[0]
 * Ambos encolados son asíncronos: quien mide debe delimitar la llamada con
 * eventos CUDA y sincronizar (lo hace main.cu). El tiempo medido cubre las
 * DOS etapas: es el coste real del algoritmo completo.
 *
 * @param deviceInput    Vector de entrada en GPU (N elementos).
 * @param devicePartials Buffer de trabajo en GPU (kReductionStage1Blocks floats).
 * @param deviceResult   Salida en GPU: 1 float con la suma total.
 * @param elementCount   Número de elementos N del vector de entrada.
 * @param stream         Stream CUDA donde encolar (0 = stream por defecto).
 */
void launchReduction(const float* deviceInput,
                     float*       devicePartials,
                     float*       deviceResult,
                     std::size_t  elementCount,
                     cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace bench
