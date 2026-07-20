/**
 * @file kernel.cpp
 * @brief Implementación del kernel OpenCL de suma de vectores —
 *        espejo de cuda/vector_add/kernel.cu.
 *
 * CONCEPTOS OPENCL QUE APARECEN AQUÍ (guía de aprendizaje):
 *
 *   __kernel     Equivalente de __global__ en CUDA: función que se ejecuta
 *                en el dispositivo, invocada desde el host. Miles de
 *                work-items la ejecutan en paralelo sobre datos distintos.
 *
 *   Jerarquía de ejecución (nomenclatura OpenCL ↔ CUDA):
 *                NDRange      ↔  grid
 *                work-group   ↔  bloque (block)
 *                work-item    ↔  hilo (thread)
 *
 *   get_global_id(0)
 *                Índice GLOBAL del work-item. OpenCL lo da directamente;
 *                en CUDA se calcula a mano: blockIdx.x*blockDim.x+threadIdx.x.
 *                Mismo valor, distinta ergonomía.
 *
 *   __global     Califica punteros a memoria global del dispositivo (la
 *                GDDR5 de la GPU) — en CUDA es implícito para los parámetros
 *                del kernel.
 */

#include "kernel.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bench {
namespace ocl {

namespace {

/**
 * @brief Work-items por work-group — MISMO VALOR que kThreadsPerBlock en CUDA.
 *
 * La equivalencia 256 = 256 es deliberada y crítica para la comparación:
 * ambos frameworks lanzan exactamente la misma geometría sobre el mismo
 * hardware (warps de 32 en Maxwell), de modo que cualquier diferencia de
 * tiempo medida sea atribuible al framework y no a la configuración.
 */
constexpr std::size_t kWorkGroupSize = 256;

/**
 * @brief Fuente OpenCL C del kernel, embebido en el binario.
 *
 * Es la traducción LITERAL del kernel CUDA (kernel.cu):
 *
 *   CUDA                                  OpenCL C
 *   ------------------------------------  -----------------------------------
 *   __global__ void vectorAddKernel(...)  __kernel void vector_add(...)
 *   const float* __restrict__ a           __global const float* restrict a
 *   blockIdx.x*blockDim.x+threadIdx.x     get_global_id(0)
 *   std::size_t elementCount              ulong element_count
 *   if (i < elementCount)                 if (i < element_count)   [la guarda]
 *   c[i] = a[i] + b[i];                   c[i] = a[i] + b[i];      [idéntico]
 *
 * La GUARDA sigue siendo obligatoria por la misma razón que en CUDA: el
 * global size se redondea hacia arriba a múltiplo de 256, y los work-items
 * sobrantes no deben tocar memoria.
 *
 * Se embebe como raw string (R"CLC(...)CLC") en lugar de cargar un archivo
 * .cl en runtime: el binario queda autocontenido y no depende de rutas
 * relativas al directorio de trabajo (robusto frente a scripts).
 */
constexpr const char* kKernelSource = R"CLC(
__kernel void vector_add(__global const float* restrict a,
                         __global const float* restrict b,
                         __global float* restrict c,
                         const ulong element_count)
{
    /* Indice global de este work-item dentro de todo el NDRange. */
    const size_t i = get_global_id(0);

    /* Guarda de limites: solo los work-items con indice valido trabajan. */
    if (i < element_count) {
        c[i] = a[i] + b[i];  /* El algoritmo completo: una suma por work-item. */
    }
}
)CLC";

}  // namespace

VectorAddKernel::VectorAddKernel(cl_context context, cl_device_id device) {
    cl_int status = CL_SUCCESS;

    // 1. Crear el objeto programa a partir del fuente embebido.
    // El API C pide const char** (sin const en el puntero exterior), asi que
    // el fuente se pasa via una variable local no-constexpr.
    const char* source = kKernelSource;
    program_ = clCreateProgramWithSource(context, 1, &source, nullptr, &status);
    CL_CHECK(status);

    // 2. Compilar para el dispositivo (compilación ONLINE, propia de OpenCL;
    //    debe ocurrir en el warm-up, nunca dentro de la zona medida).
    //    Sin flags de fast-math: misma semántica IEEE 754 que la build CUDA.
    const cl_int buildStatus = clBuildProgram(program_, 1, &device, "", nullptr, nullptr);

    if (buildStatus != CL_SUCCESS) {
        // El log del compilador del driver es la única pista útil ante un
        // error de OpenCL C: se imprime completo antes de abortar.
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

    // 3. Extraer el kernel por nombre (debe coincidir con el del fuente).
    kernel_ = clCreateKernel(program_, "vector_add", &status);
    CL_CHECK(status);
}

VectorAddKernel::~VectorAddKernel() {
    if (kernel_)  clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
}

void VectorAddKernel::enqueue(cl_command_queue queue,
                              cl_mem deviceA,
                              cl_mem deviceB,
                              cl_mem deviceC,
                              std::size_t elementCount,
                              cl_event* outEvent) const {
    // Argumentos del kernel. A diferencia de CUDA (paso directo en <<<>>>),
    // OpenCL los fija uno a uno por índice — más verboso, mismo efecto.
    const cl_ulong count = static_cast<cl_ulong>(elementCount);
    CL_CHECK(clSetKernelArg(kernel_, 0, sizeof(cl_mem), &deviceA));
    CL_CHECK(clSetKernelArg(kernel_, 1, sizeof(cl_mem), &deviceB));
    CL_CHECK(clSetKernelArg(kernel_, 2, sizeof(cl_mem), &deviceC));
    CL_CHECK(clSetKernelArg(kernel_, 3, sizeof(cl_ulong), &count));

    // Geometría del NDRange — el MISMO ceil-div que en CUDA:
    // global = ceil(N / 256) * 256. OpenCL pide el total de work-items
    // (global size), no el número de grupos como CUDA; el redondeo es el
    // mismo y los sobrantes los descarta la guarda del kernel.
    const std::size_t globalSize =
        ((elementCount + kWorkGroupSize - 1) / kWorkGroupSize) * kWorkGroupSize;
    const std::size_t localSize = kWorkGroupSize;

    // Encolado ASÍNCRONO (espejo del lanzamiento <<<...>>>). El evento
    // devuelto registra las marcas de tiempo de la ejecución real.
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_,
                                    1,            // dimensiones del NDRange
                                    nullptr,      // offset global (ninguno)
                                    &globalSize, &localSize,
                                    0, nullptr,   // sin dependencias previas
                                    outEvent));
}

}  // namespace ocl
}  // namespace bench
