/**
 * @file kernel.cuh
 * @brief Interfaz pública del kernel de suma de vectores (CUDA).
 *
 * DECISIÓN DE DISEÑO — OCULTAR EL LANZAMIENTO:
 * main.cu NO lanza el kernel directamente con la sintaxis <<<...>>>.
 * En su lugar llama a esta función wrapper, que encapsula:
 *   - la elección del tamaño de bloque,
 *   - el cálculo de la geometría del grid,
 *   - la comprobación de errores de lanzamiento.
 *
 * Ventajas:
 *   1. main.cu queda como un orquestador legible, sin detalles de bajo nivel.
 *   2. Si mañana cambiamos la configuración de lanzamiento (p. ej. tamaño de
 *      bloque), solo se toca kernel.cu — nada más se recompila conceptualmente.
 *   3. Es el mismo patrón que usaremos en matrix_mul y reduction, y el
 *      espejo natural de clEnqueueNDRangeKernel en la fase OpenCL.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace bench {
namespace cuda {

/**
 * @brief Lanza el kernel C[i] = A[i] + B[i] sobre memoria YA residente en GPU.
 *
 * Esta función solo ENCOLA el lanzamiento (es asíncrona): quien mide el
 * tiempo debe delimitarla con eventos CUDA y sincronizar (lo hace main.cu).
 *
 * @param deviceA      Puntero a A en memoria de dispositivo (entrada).
 * @param deviceB      Puntero a B en memoria de dispositivo (entrada).
 * @param deviceC      Puntero a C en memoria de dispositivo (salida).
 * @param elementCount Número de elementos de cada vector.
 * @param stream       Stream CUDA donde encolar (0 = stream por defecto).
 */
void launchVectorAdd(const float* deviceA,
                     const float* deviceB,
                     float*       deviceC,
                     std::size_t  elementCount,
                     cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace bench
