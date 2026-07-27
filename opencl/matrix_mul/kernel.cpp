/**
 * @file kernel.cpp
 * @brief Implementación del kernel OpenCL de multiplicación de matrices —
 *        espejo de cuda/matrix_mul/kernel.cu.
 *
 * CONCEPTOS OPENCL NUEVOS RESPECTO DE vector_add (nomenclatura ↔ CUDA):
 *
 *   __local                       ↔  __shared__ (memoria en chip por work-group)
 *   barrier(CLK_LOCAL_MEM_FENCE)  ↔  __syncthreads()
 *   get_local_id(0/1)             ↔  threadIdx.x / threadIdx.y
 *   get_group_id(0/1)             ↔  blockIdx.x / blockIdx.y
 *   NDRange 2D                    ↔  grid 2D (dim3)
 *
 * El algoritmo (tiling 16×16 con carga cooperativa y dos barreras por
 * baldosa) es la traducción LITERAL del kernel CUDA: mismas fases, mismas
 * guardas, mismo orden de acumulación sobre k. La comparación entre
 * frameworks queda así libre de diferencias algorítmicas.
 */

#include "kernel.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace bench {
namespace ocl {

namespace {

/// Lado de la baldosa y del work-group: 16×16 = 256 work-items — MISMO valor
/// que kTileSize en CUDA (geometría idéntica sobre el mismo hardware).
constexpr std::size_t kTileSize = 16;

/**
 * @brief Fuente OpenCL C del kernel, embebido en el binario.
 *
 * Traducción literal de matrixMulKernel (cuda/matrix_mul/kernel.cu):
 * misma estructura de 4 fases por baldosa (carga cooperativa → barrera →
 * cómputo → barrera), mismas guardas con carga de 0 (neutro de la suma)
 * para mantener las baldosas completas y las barreras uniformes.
 *
 * NOTA sobre el orden de acumulación: k asciende de 0 a N-1 igual que en
 * CUDA y que en la referencia CPU. Las diferencias residuales (contracción
 * FMA de cada compilador) las cubre la tolerancia de la verificación.
 */
constexpr const char* kKernelSource = R"CLC(
#define TILE 16

__kernel void matrix_mul(__global const float* restrict a,
                         __global const float* restrict b,
                         __global float* restrict c,
                         const uint n)
{
    /* Baldosas en memoria local (el __shared__ de OpenCL). */
    __local float tileA[TILE][TILE];
    __local float tileB[TILE][TILE];

    /* Coordenadas del elemento de C asignado a este work-item.
       Dimension 0 recorre columnas (memoria contigua => coalescido). */
    const uint lx  = get_local_id(0);
    const uint ly  = get_local_id(1);
    const uint col = get_group_id(0) * TILE + lx;
    const uint row = get_group_id(1) * TILE + ly;

    float acc = 0.0f;  /* Acumulador del producto escalar (registro privado). */

    const uint numTiles = (n + TILE - 1) / TILE;
    for (uint t = 0; t < numTiles; ++t) {
        /* Fase 1: carga cooperativa de las baldosas t de A y B. */
        const uint aCol = t * TILE + lx;
        const uint bRow = t * TILE + ly;

        tileA[ly][lx] = (row < n && aCol < n) ? a[(size_t)row * n + aCol] : 0.0f;
        tileB[ly][lx] = (bRow < n && col < n) ? b[(size_t)bRow * n + col] : 0.0f;

        /* Fase 2: barrera — la baldosa debe estar completa. */
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Fase 3: producto parcial usando solo memoria local. */
        for (uint k = 0; k < TILE; ++k) {
            acc += tileA[ly][k] * tileB[k][lx];
        }

        /* Fase 4: barrera — nadie recarga hasta que todos leyeron. */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Solo los work-items con coordenadas validas escriben su elemento. */
    if (row < n && col < n) {
        c[(size_t)row * n + col] = acc;
    }
}
)CLC";

}  // namespace

MatrixMulKernel::MatrixMulKernel(cl_context context, cl_device_id device) {
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
    kernel_ = clCreateKernel(program_, "matrix_mul", &status);
    CL_CHECK(status);
}

MatrixMulKernel::~MatrixMulKernel() {
    if (kernel_)  clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
}

void MatrixMulKernel::enqueue(cl_command_queue queue,
                              cl_mem deviceA,
                              cl_mem deviceB,
                              cl_mem deviceC,
                              std::size_t dimension,
                              cl_event* outEvent) const {
    const cl_uint n = static_cast<cl_uint>(dimension);
    CL_CHECK(clSetKernelArg(kernel_, 0, sizeof(cl_mem), &deviceA));
    CL_CHECK(clSetKernelArg(kernel_, 1, sizeof(cl_mem), &deviceB));
    CL_CHECK(clSetKernelArg(kernel_, 2, sizeof(cl_mem), &deviceC));
    CL_CHECK(clSetKernelArg(kernel_, 3, sizeof(cl_uint), &n));

    // NDRange 2D — mismo ceil-div que el grid 2D de CUDA, expresado como
    // total de work-items por dimensión (global = grupos × 16).
    const std::size_t groupsPerSide = (dimension + kTileSize - 1) / kTileSize;
    const std::size_t globalSize[2] = {groupsPerSide * kTileSize,
                                       groupsPerSide * kTileSize};
    const std::size_t localSize[2]  = {kTileSize, kTileSize};

    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_,
                                    2,            // NDRange bidimensional
                                    nullptr,      // sin offset global
                                    globalSize, localSize,
                                    0, nullptr,   // sin dependencias previas
                                    outEvent));
}

}  // namespace ocl
}  // namespace bench
