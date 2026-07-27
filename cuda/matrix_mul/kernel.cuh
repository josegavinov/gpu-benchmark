/**
 * @file kernel.cuh
 * @brief Interfaz pública del kernel de multiplicación de matrices (CUDA).
 *
 * MISMA DECISIÓN DE DISEÑO QUE EN vector_add — OCULTAR EL LANZAMIENTO:
 * main.cu no usa la sintaxis <<<...>>>; este wrapper encapsula la geometría
 * del grid (aquí BIDIMENSIONAL: un hilo por elemento de C) y la comprobación
 * de errores de lanzamiento.
 *
 * CONVENCIÓN DEL BENCHMARK: matrices CUADRADAS de dimensión N×N, almacenadas
 * por filas (row-major) en un buffer lineal: A[fila][col] = A[fila*N + col].
 * ProblemSize (la N de la línea de comandos) es la DIMENSIÓN de la matriz,
 * no el número total de elementos (que es N²).
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace bench {
namespace cuda {

/**
 * @brief Lanza el kernel C = A × B sobre memoria YA residente en GPU.
 *
 * Solo ENCOLA el lanzamiento (asíncrono): quien mide el tiempo debe
 * delimitarlo con eventos CUDA y sincronizar (lo hace main.cu).
 *
 * @param deviceA   Matriz A (N×N, row-major) en memoria de dispositivo.
 * @param deviceB   Matriz B (N×N, row-major) en memoria de dispositivo.
 * @param deviceC   Matriz C (N×N, row-major) de salida en dispositivo.
 * @param dimension Dimensión N de las matrices cuadradas.
 * @param stream    Stream CUDA donde encolar (0 = stream por defecto).
 */
void launchMatrixMul(const float* deviceA,
                     const float* deviceB,
                     float*       deviceC,
                     std::size_t  dimension,
                     cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace bench
