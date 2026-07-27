/**
 * @file kernel.cpp
 * @brief Implementación del kernel OpenCL de reducción —
 *        espejo de cuda/reduction/kernel.cu.
 *
 * El algoritmo es la traducción LITERAL del kernel CUDA (mismas tres fases:
 * acumulación grid-stride → árbol en memoria local → escritura de la
 * parcial), con la misma geometría fija de 256 grupos × 256 work-items y la
 * misma estrategia de dos etapas reutilizando un único kernel.
 *
 * Correspondencias (además de las ya vistas en matrix_mul):
 *
 *   get_global_size(0)  ↔  gridDim.x * blockDim.x   (el paso del grid-stride)
 *   get_local_size(0)   ↔  blockDim.x
 *
 * DETALLE DE API: los argumentos de un cl_kernel se fijan con clSetKernelArg
 * ANTES de cada encolado, y el encolado los "captura" en ese momento. Por
 * eso es válido reutilizar el mismo cl_kernel para las dos etapas cambiando
 * los argumentos entre ambas — el espejo de pasar parámetros distintos a los
 * dos lanzamientos <<<...>>> de CUDA.
 */

#include "kernel.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bench {
namespace ocl {

namespace {

/// Work-items por work-group: mismo valor (256, potencia de 2) que
/// kThreadsPerBlock en la versión CUDA — geometría idéntica.
constexpr std::size_t kWorkGroupSize = 256;

/**
 * @brief Fuente OpenCL C del kernel, embebido en el binario.
 *
 * Traducción literal de reduceSumKernel (cuda/reduction/kernel.cu):
 *
 *   Fase 1: acumulación secuencial con paso de grid (get_global_size(0)),
 *           accesos coalescidos, sin sincronización (registros privados).
 *   Fase 2: volcado a memoria local + reducción en árbol con
 *           barrier(CLK_LOCAL_MEM_FENCE) entre pasos (↔ __syncthreads()).
 *   Fase 3: el work-item 0 escribe la parcial de su grupo.
 *
 * El tamaño del array __local (256) debe coincidir con el local size del
 * encolado — igual que el __shared__[256] de CUDA con blockDim 256.
 */
constexpr const char* kKernelSource = R"CLC(
#define WG_SIZE 256

__kernel void reduce_sum(__global const float* restrict input,
                         __global float* restrict partials,
                         const ulong element_count)
{
    __local float scratch[WG_SIZE];

    const size_t tid = get_local_id(0);

    /* Fase 1: acumulacion secuencial con paso de grid. */
    const size_t gridStride = get_global_size(0);
    float acc = 0.0f;
    for (size_t i = get_global_id(0); i < element_count; i += gridStride) {
        acc += input[i];
    }

    /* Fase 2: reduccion en arbol dentro del work-group. */
    scratch[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);  /* Todos los acumuladores volcados. */

    for (size_t s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  /* El paso siguiente lee este. */
    }

    /* Fase 3: una suma parcial por work-group. */
    if (tid == 0) {
        partials[get_group_id(0)] = scratch[0];
    }
}
)CLC";

/// Fija los argumentos y encola UNA etapa del kernel de reducción.
void enqueueStage(cl_command_queue queue,
                  cl_kernel kernel,
                  cl_mem input,
                  cl_mem output,
                  std::size_t elementCount,
                  std::size_t groupCount,
                  cl_event* outEvent) {
    const cl_ulong count = static_cast<cl_ulong>(elementCount);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &input));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &output));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_ulong), &count));

    const std::size_t globalSize = groupCount * kWorkGroupSize;
    const std::size_t localSize  = kWorkGroupSize;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel,
                                    1, nullptr,
                                    &globalSize, &localSize,
                                    0, nullptr, outEvent));
}

}  // namespace

ReductionKernel::ReductionKernel(cl_context context, cl_device_id device) {
    cl_int status = CL_SUCCESS;

    // 1. Crear el objeto programa a partir del fuente embebido.
    const char* source = kKernelSource;
    program_ = clCreateProgramWithSource(context, 1, &source, nullptr, &status);
    CL_CHECK(status);

    // 2. Compilación ONLINE (en el warm-up, nunca en la zona medida).
    //    Sin flags de fast-math: misma semántica IEEE 754 que la build CUDA.
    const cl_int buildStatus = clBuildProgram(program_, 1, &device, "", nullptr, nullptr);

    if (buildStatus != CL_SUCCESS) {
        std::size_t logSize = 0;
        clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &logSize);
        std::vector<char> log(logSize + 1, '\0');
        clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                              logSize, log.data(), nullptr);
        std::fprintf(stderr,
                     "[OpenCL ERROR] Fallo la compilacion del kernel:\n%s\n",
                     log.data());
        std::exit(EXIT_FAILURE);
    }

    // 3. Extraer el kernel por nombre.
    kernel_ = clCreateKernel(program_, "reduce_sum", &status);
    CL_CHECK(status);
}

ReductionKernel::~ReductionKernel() {
    if (kernel_)  clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
}

void ReductionKernel::enqueue(cl_command_queue queue,
                              cl_mem deviceInput,
                              cl_mem devicePartials,
                              cl_mem deviceResult,
                              std::size_t elementCount,
                              cl_event* outEventStage1,
                              cl_event* outEventStage2) const {
    // Etapa 1: N elementos → kReductionStage1Groups sumas parciales.
    enqueueStage(queue, kernel_, deviceInput, devicePartials,
                 elementCount, kReductionStage1Groups, outEventStage1);

    // Etapa 2: parciales → suma total, con UN solo work-group. La cola es
    // in-order: la etapa 2 espera a la 1 sin intervención de la CPU.
    enqueueStage(queue, kernel_, devicePartials, deviceResult,
                 kReductionStage1Groups, 1, outEventStage2);
}

}  // namespace ocl
}  // namespace bench
