/**
 * @file kernel.hpp
 * @brief Interfaz pública del kernel de reducción (OpenCL) —
 *        espejo de cuda/reduction/kernel.cuh.
 *
 * MISMO PATRÓN QUE LOS DEMÁS BENCHMARKS OPENCL: la clase posee cl_program y
 * cl_kernel (compilación online en el constructor), y enqueue() encapsula la
 * estrategia completa de DOS encolados encadenados — el espejo exacto de los
 * dos lanzamientos de launchReduction() en CUDA (la justificación de las dos
 * etapas está en cuda/reduction/kernel.cu y aplica igual aquí: barrier()
 * solo sincroniza dentro de un work-group).
 */

#pragma once

#include "opencl/common/opencl_utils.hpp"

#include <cstddef>

namespace bench {
namespace ocl {

/// Work-groups de la primera etapa — MISMO valor que kReductionStage1Blocks
/// en CUDA. main.cpp lo usa para dimensionar el buffer de sumas parciales.
inline constexpr std::size_t kReductionStage1Groups = 256;

/**
 * @brief Kernel de reducción ya compilado y listo para encolar.
 */
class ReductionKernel {
public:
    /**
     * @brief Compila el fuente OpenCL C y crea el objeto kernel.
     */
    ReductionKernel(cl_context context, cl_device_id device);

    ~ReductionKernel();

    ReductionKernel(const ReductionKernel&)            = delete;
    ReductionKernel& operator=(const ReductionKernel&) = delete;

    /**
     * @brief Encola la reducción completa: suma de N floats → 1 float.
     *
     * Dos encolados asíncronos a la MISMA cola (in-order: la etapa 2 no
     * arranca hasta terminar la 1, sin sincronizar con la CPU — el espejo
     * del stream CUDA). Devuelve un evento por etapa: Kernel_ms es la suma
     * de ambos intervalos de profiling, el coste del algoritmo completo.
     *
     * @param queue          Cola de comandos (con profiling habilitado).
     * @param deviceInput    Vector de entrada en GPU (N elementos).
     * @param devicePartials Buffer de trabajo (kReductionStage1Groups floats).
     * @param deviceResult   Salida en GPU: 1 float con la suma total.
     * @param elementCount   Número de elementos N del vector de entrada.
     * @param outEventStage1 Evento del primer encolado (para profiling).
     * @param outEventStage2 Evento del segundo encolado (para profiling).
     */
    void enqueue(cl_command_queue queue,
                 cl_mem deviceInput,
                 cl_mem devicePartials,
                 cl_mem deviceResult,
                 std::size_t elementCount,
                 cl_event* outEventStage1,
                 cl_event* outEventStage2) const;

private:
    cl_program program_ = nullptr;
    cl_kernel  kernel_  = nullptr;
};

}  // namespace ocl
}  // namespace bench
