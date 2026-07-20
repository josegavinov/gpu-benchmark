/**
 * @file kernel.hpp
 * @brief Interfaz pública del kernel de suma de vectores (OpenCL) —
 *        espejo de cuda/vector_add/kernel.cuh.
 *
 * MISMA DECISIÓN DE DISEÑO QUE EN CUDA — OCULTAR EL LANZAMIENTO:
 * main.cpp no llama a clEnqueueNDRangeKernel directamente. Este módulo
 * encapsula:
 *   - el código fuente OpenCL C del kernel (embebido en el binario),
 *   - la compilación en runtime (clBuildProgram) con diagnóstico de errores,
 *   - la elección del tamaño de work-group y el redondeo del global size,
 *   - el encolado del kernel.
 *
 * DIFERENCIA CLAVE CON CUDA (objeto de estudio): CUDA compila el kernel
 * OFFLINE (nvcc genera código sm_50 dentro del ejecutable); OpenCL compila
 * ONLINE (el driver compila el fuente al arrancar el programa). Por eso
 * existe buildVectorAddKernel(): esa compilación ocurre UNA vez, en el
 * warm-up, fuera de toda medición — simétrico a la carga de módulo que el
 * warm-up de CUDA también excluye.
 */

#pragma once

#include "opencl/common/opencl_utils.hpp"

#include <cstddef>

namespace bench {
namespace ocl {

/**
 * @brief Kernel de suma de vectores ya compilado y listo para encolar.
 *
 * Posee el cl_program y el cl_kernel; los libera en el destructor (RAII,
 * como ClEnvironment). Se construye una sola vez por proceso.
 */
class VectorAddKernel {
public:
    /**
     * @brief Compila el fuente OpenCL C y crea el objeto kernel.
     *
     * Si la compilación falla, imprime el LOG COMPLETO del compilador del
     * driver (la única forma de diagnosticar errores de OpenCL C) y termina.
     */
    VectorAddKernel(cl_context context, cl_device_id device);

    ~VectorAddKernel();

    VectorAddKernel(const VectorAddKernel&)            = delete;
    VectorAddKernel& operator=(const VectorAddKernel&) = delete;

    /**
     * @brief Encola una ejecución C[i] = A[i] + B[i] sobre buffers de GPU.
     *
     * Solo ENCOLA (asíncrono), igual que launchVectorAdd en CUDA. El evento
     * devuelto en outEvent permite medir el tiempo real de ejecución con
     * eventElapsedMs() — el espejo del EventTimer alrededor del kernel CUDA.
     *
     * @param queue        Cola de comandos (con profiling habilitado).
     * @param deviceA      Buffer de A en GPU (entrada).
     * @param deviceB      Buffer de B en GPU (entrada).
     * @param deviceC      Buffer de C en GPU (salida).
     * @param elementCount Número de elementos de cada vector.
     * @param outEvent     Evento del comando encolado (para profiling).
     */
    void enqueue(cl_command_queue queue,
                 cl_mem deviceA,
                 cl_mem deviceB,
                 cl_mem deviceC,
                 std::size_t elementCount,
                 cl_event* outEvent) const;

private:
    cl_program program_ = nullptr;
    cl_kernel  kernel_  = nullptr;
};

}  // namespace ocl
}  // namespace bench
