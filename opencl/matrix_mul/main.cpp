/**
 * @file main.cpp
 * @brief Benchmark Matrix Multiplication (OpenCL) — orquestador.
 *        ESPEJO 1:1 de cuda/matrix_mul/main.cu (mismas 8 fases).
 *
 * PUNTO METODOLÓGICO CENTRAL (idéntico a vector_add): las fases de
 * argumentos, datos, verificación y CSV usan EXACTAMENTE el mismo código
 * compartido de include/common/ que la versión CUDA. Mismas semillas →
 * mismas matrices → misma referencia CPU → misma tolerancia (1e-3, cuya
 * justificación numérica está en cuda/matrix_mul/main.cu). Las dos
 * implementaciones solo difieren en el código GPU.
 */

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "include/common/benchmark_config.hpp"
#include "include/common/csv_logger.hpp"
#include "include/common/data_init.hpp"
#include "include/common/verification.hpp"

#include "opencl/common/opencl_utils.hpp"

#include "kernel.hpp"

namespace {

// ---------------------------------------------------------------------------
// Constantes de identificación del experimento (columnas fijas del CSV).
// ---------------------------------------------------------------------------
constexpr const char* kFramework      = "OpenCL";
constexpr const char* kBenchmarkName  = "MatrixMultiplication";
constexpr const char* kDefaultCsvPath = "results/opencl/matrix_mul.csv";

/// Tolerancia relativa — el MISMO valor que la versión CUDA (misma vara).
constexpr float kRelativeTolerance = 1e-3f;

// ---------------------------------------------------------------------------
// Tipos auxiliares
// ---------------------------------------------------------------------------

/// Buffers de las tres matrices en memoria de la GPU (cl_mem ↔ float* CUDA).
struct DeviceBuffers {
    cl_mem a = nullptr;
    cl_mem b = nullptr;
    cl_mem c = nullptr;
};

/// Los cuatro tiempos que exige el protocolo, en milisegundos.
struct Timings {
    double hostToDeviceMs = 0.0;
    double kernelMs       = 0.0;
    double deviceToHostMs = 0.0;
    double totalMs        = 0.0;
};

// ---------------------------------------------------------------------------
// Fase 2: inicialización de datos — IDÉNTICA a la versión CUDA (mismo código,
// mismas semillas, mismo orden i,k,j de la referencia; ver justificación en
// cuda/matrix_mul/main.cu).
// ---------------------------------------------------------------------------
void initializeHostData(std::size_t dimension,
                        std::vector<float>& hostA,
                        std::vector<float>& hostB,
                        std::vector<float>& cpuReference) {
    const std::size_t elementCount = dimension * dimension;
    bench::fillRandom(hostA, elementCount, bench::kSeedVectorA);
    bench::fillRandom(hostB, elementCount, bench::kSeedVectorB);

    cpuReference.assign(elementCount, 0.0f);
    for (std::size_t i = 0; i < dimension; ++i) {
        for (std::size_t k = 0; k < dimension; ++k) {
            const float aik = hostA[i * dimension + k];
            for (std::size_t j = 0; j < dimension; ++j) {
                cpuReference[i * dimension + j] += aik * hostB[k * dimension + j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fase 3: calentamiento — simétrico al warmUpGpu() de la versión CUDA.
// Contexto, cola y COMPILACIÓN ONLINE ya ocurrieron al construir env/kernel;
// aquí se añade el primer encolado real (mini-multiplicación de 64×64).
// ---------------------------------------------------------------------------
void warmUpGpu(const bench::ocl::ClEnvironment& env,
               const bench::ocl::MatrixMulKernel& kernel) {
    constexpr std::size_t warmupDim   = 64;
    constexpr std::size_t warmupBytes = warmupDim * warmupDim * sizeof(float);

    cl_int status = CL_SUCCESS;
    cl_mem scratch = clCreateBuffer(env.context(), CL_MEM_READ_WRITE,
                                    warmupBytes, nullptr, &status);
    CL_CHECK(status);

    cl_event event = nullptr;
    kernel.enqueue(env.queue(), scratch, scratch, scratch, warmupDim, &event);
    CL_CHECK(clWaitForEvents(1, &event));   // Esperar a que termine de verdad.
    CL_CHECK(clReleaseEvent(event));
    CL_CHECK(clReleaseMemObject(scratch));
}

// ---------------------------------------------------------------------------
// Fases 4 y 8: gestión de memoria de dispositivo
// ---------------------------------------------------------------------------

/// Reserva las tres matrices N×N en la GPU (flags según el uso real).
DeviceBuffers allocateDevice(cl_context context, std::size_t dimension) {
    const std::size_t bytes = dimension * dimension * sizeof(float);
    DeviceBuffers buffers;
    cl_int status = CL_SUCCESS;

    buffers.a = clCreateBuffer(context, CL_MEM_READ_ONLY,  bytes, nullptr, &status);
    CL_CHECK(status);
    buffers.b = clCreateBuffer(context, CL_MEM_READ_ONLY,  bytes, nullptr, &status);
    CL_CHECK(status);
    buffers.c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &status);
    CL_CHECK(status);
    return buffers;
}

/// Libera los buffers de GPU. Se llama SIEMPRE, incluso si la verificación falla.
void freeDevice(DeviceBuffers& buffers) {
    if (buffers.a) CL_CHECK(clReleaseMemObject(buffers.a));
    if (buffers.b) CL_CHECK(clReleaseMemObject(buffers.b));
    if (buffers.c) CL_CHECK(clReleaseMemObject(buffers.c));
    buffers = DeviceBuffers{};  // Evitar handles colgantes.
}

// ---------------------------------------------------------------------------
// Fase 5: pipeline medido (H2D → kernel → D2H) — mismo patrón que vector_add
// ---------------------------------------------------------------------------

/// Copia A y B del host a la GPU, midiendo con eventos de profiling. Devuelve ms.
double copyHostToDevice(cl_command_queue queue,
                        const std::vector<float>& hostA,
                        const std::vector<float>& hostB,
                        const DeviceBuffers& buffers) {
    const std::size_t bytes = hostA.size() * sizeof(float);

    cl_event eventA = nullptr;
    cl_event eventB = nullptr;
    CL_CHECK(clEnqueueWriteBuffer(queue, buffers.a, CL_TRUE, 0, bytes,
                                  hostA.data(), 0, nullptr, &eventA));
    CL_CHECK(clEnqueueWriteBuffer(queue, buffers.b, CL_TRUE, 0, bytes,
                                  hostB.data(), 0, nullptr, &eventB));

    const double ms = bench::ocl::eventElapsedMs(eventA) +
                      bench::ocl::eventElapsedMs(eventB);
    CL_CHECK(clReleaseEvent(eventA));
    CL_CHECK(clReleaseEvent(eventB));
    return ms;
}

/// Ejecuta el kernel una vez, midiendo SOLO su ejecución. Devuelve ms.
double runKernel(cl_command_queue queue,
                 const bench::ocl::MatrixMulKernel& kernel,
                 const DeviceBuffers& buffers,
                 std::size_t dimension) {
    cl_event event = nullptr;
    kernel.enqueue(queue, buffers.a, buffers.b, buffers.c, dimension, &event);
    const double ms = bench::ocl::eventElapsedMs(event);
    CL_CHECK(clReleaseEvent(event));
    return ms;
}

/// Copia el resultado C de la GPU al host, midiendo con eventos. Devuelve ms.
double copyDeviceToHost(cl_command_queue queue,
                        std::vector<float>& hostC,
                        const DeviceBuffers& buffers) {
    const std::size_t bytes = hostC.size() * sizeof(float);

    cl_event event = nullptr;
    CL_CHECK(clEnqueueReadBuffer(queue, buffers.c, CL_TRUE, 0, bytes,
                                 hostC.data(), 0, nullptr, &event));
    const double ms = bench::ocl::eventElapsedMs(event);
    CL_CHECK(clReleaseEvent(event));
    return ms;
}

/// Pipeline completo H2D → kernel → D2H; Total_ms con reloj monotónico de host.
Timings runMeasuredPipeline(cl_command_queue queue,
                            const bench::ocl::MatrixMulKernel& kernel,
                            const std::vector<float>& hostA,
                            const std::vector<float>& hostB,
                            std::vector<float>& hostC,
                            const DeviceBuffers& buffers,
                            std::size_t dimension) {
    Timings timings;

    const auto totalStart = std::chrono::steady_clock::now();

    timings.hostToDeviceMs = copyHostToDevice(queue, hostA, hostB, buffers);
    timings.kernelMs       = runKernel(queue, kernel, buffers, dimension);
    timings.deviceToHostMs = copyDeviceToHost(queue, hostC, buffers);

    const auto totalEnd = std::chrono::steady_clock::now();
    timings.totalMs =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    return timings;
}

}  // namespace

// ---------------------------------------------------------------------------
// main: solo coordina; misma estructura y orden que cuda/matrix_mul/main.cu.
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // ---- 1. Argumentos de línea de comandos -------------------------------
    const auto config = bench::parseArgs(argc, argv, kDefaultCsvPath);
    if (!config) {
        return EXIT_FAILURE;  // parseArgs ya explicó el problema por stderr.
    }

    // ---- 2. Datos del host (reproducibles) y referencia CPU ---------------
    std::vector<float> hostA;
    std::vector<float> hostB;
    std::vector<float> cpuReference;
    initializeHostData(config->problemSize, hostA, hostB, cpuReference);

    // Buffer de resultados prellenado a cero (misma razón que en CUDA).
    std::vector<float> hostC(config->problemSize * config->problemSize, 0.0f);

    // ---- 3. Entorno OpenCL + calentamiento --------------------------------
    // Construir env y kernel YA ES parte del warm-up: contexto, cola y
    // compilación online ocurren aquí, fuera de la zona medida.
    bench::ocl::ClEnvironment env;
    bench::ocl::printDeviceInfo(env.device());
    std::cerr << "[INFO] N = " << config->problemSize << " (matriz "
              << config->problemSize << "x" << config->problemSize << ")"
              << " | iteracion = " << config->iteration
              << " | csv = " << config->csvPath << '\n';

    bench::ocl::MatrixMulKernel kernel(env.context(), env.device());
    warmUpGpu(env, kernel);

    // ---- 4. Memoria de dispositivo ----------------------------------------
    DeviceBuffers buffers = allocateDevice(env.context(), config->problemSize);

    // ---- 5. Pipeline medido ------------------------------------------------
    const Timings timings = runMeasuredPipeline(
        env.queue(), kernel, hostA, hostB, hostC, buffers, config->problemSize);

    // ---- 6. Verificación funcional (100% de los N² elementos) --------------
    const bench::VerificationResult verdict = bench::verifyElementwise(
        cpuReference.data(), hostC.data(), hostC.size(), kRelativeTolerance);
    bench::printVerification(verdict);

    // ---- 7. Registro en CSV (siempre, también si es FAIL) ------------------
    bench::BenchmarkResult result;
    result.framework      = kFramework;
    result.benchmark      = kBenchmarkName;
    result.problemSize    = config->problemSize;
    result.iteration      = config->iteration;
    result.hostToDeviceMs = timings.hostToDeviceMs;
    result.kernelMs       = timings.kernelMs;
    result.deviceToHostMs = timings.deviceToHostMs;
    result.totalMs        = timings.totalMs;
    result.status         = verdict.status();

    try {
        bench::CsvLogger(config->csvPath).append(result);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        freeDevice(buffers);
        return EXIT_FAILURE;
    }

    // ---- 8. Liberación de recursos -----------------------------------------
    // Buffers explícitos aquí; contexto, cola, programa y kernel se liberan
    // solos al destruirse env y kernel (RAII).
    freeDevice(buffers);

    // Código de salida útil para los scripts: 0 solo si la verificación pasó.
    return verdict.passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
