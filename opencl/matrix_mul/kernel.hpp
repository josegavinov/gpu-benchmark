/**
 * @file kernel.hpp
 * @brief Interfaz pública del kernel de multiplicación de matrices (OpenCL) —
 *        espejo de cuda/matrix_mul/kernel.cuh.
 *
 * MISMO PATRÓN QUE opencl/vector_add: la clase posee el cl_program y el
 * cl_kernel (compilación ONLINE en el constructor, fuera de la zona medida),
 * y enqueue() encapsula la geometría del NDRange — aquí BIDIMENSIONAL, el
 * espejo del grid 2D de la versión CUDA.
 *
 * CONVENCIÓN DEL BENCHMARK: matrices CUADRADAS N×N row-major (idéntica a la
 * versión CUDA; ver cuda/matrix_mul/kernel.cuh).
 */

#pragma once

#include "opencl/common/opencl_utils.hpp"

#include <cstddef>

namespace bench {
namespace ocl {

/**
 * @brief Kernel de multiplicación de matrices ya compilado y listo para encolar.
 */
class MatrixMulKernel {
public:
    /**
     * @brief Compila el fuente OpenCL C y crea el objeto kernel.
     *
     * Si la compilación falla, imprime el log completo del compilador del
     * driver y termina (misma política que VectorAddKernel).
     */
    MatrixMulKernel(cl_context context, cl_device_id device);

    ~MatrixMulKernel();

    MatrixMulKernel(const MatrixMulKernel&)            = delete;
    MatrixMulKernel& operator=(const MatrixMulKernel&) = delete;

    /**
     * @brief Encola una ejecución C = A × B sobre buffers de GPU.
     *
     * Solo ENCOLA (asíncrono). El evento devuelto permite medir el tiempo
     * real de ejecución con eventElapsedMs().
     *
     * @param queue     Cola de comandos (con profiling habilitado).
     * @param deviceA   Matriz A (N×N, row-major) en GPU.
     * @param deviceB   Matriz B (N×N, row-major) en GPU.
     * @param deviceC   Matriz C (N×N, row-major) de salida en GPU.
     * @param dimension Dimensión N de las matrices cuadradas.
     * @param outEvent  Evento del comando encolado (para profiling).
     */
    void enqueue(cl_command_queue queue,
                 cl_mem deviceA,
                 cl_mem deviceB,
                 cl_mem deviceC,
                 std::size_t dimension,
                 cl_event* outEvent) const;

private:
    cl_program program_ = nullptr;
    cl_kernel  kernel_  = nullptr;
};

}  // namespace ocl
}  // namespace bench
