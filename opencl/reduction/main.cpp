/**
 * @file main.cpp
 * @brief Benchmark Reduction (OpenCL) — orquestador.
 *        ESPEJO 1:1 de cuda/reduction/main.cu (mismas 8 fases).
 *
 * Misma semántica que la versión CUDA: un solo operando de entrada,
 * resultado escalar (D2H de 4 bytes), referencia CPU acumulada en DOUBLE y
 * tolerancia relativa 1e-4 (justificación numérica en cuda/reduction/main.cu).
 *
 * Kernel_ms = suma de los intervalos de profiling de las DOS etapas — el
 * criterio análogo al EventTimer de CUDA que envuelve ambos lanzamientos,
 * y el mismo patrón "suma de eventos" que copyHostToDevice usa en
 * vector_add para las dos transferencias.
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
constexpr const char* kBenchmarkName  = "Reduction";
constexpr const char* kDefaultCsvPath = "results/opencl/reduction.csv";

/// Tolerancia relativa — el MISMO valor que la versión CUDA (misma vara).
constexpr float kRelativeTolerance = 1e-4f;

// ---------------------------------------------------------------------------
// Tipos auxiliares
// ---------------------------------------------------------------------------

/// Buffers en GPU: entrada, parciales de la etapa 1 y resultado escalar.
struct DeviceBuffers {
    cl_mem input    = nullptr;  ///< Vector de entrada (N elementos).
    cl_mem partials = nullptr;  ///< Sumas parciales (una por work-group).
    cl_mem result   = nullptr;  ///< Suma total (1 elemento).
};

/// Los cuatro tiempos que exige el protocolo, en milisegundos.
struct Timings {
    double hostToDeviceMs = 0.0;
    double kernelMs       = 0.0;
    double deviceToHostMs = 0.0;
    double totalMs        = 0.0;
};

// ---------------------------------------------------------------------------
// Fase 2: inicialización de datos — IDÉNTICA a la versión CUDA (mismo
// código, misma semilla, misma referencia en double).
// ---------------------------------------------------------------------------
void initializeHostData(std::size_t elementCount,
                        std::vector<float>& hostInput,
                        float& cpuReference) {
    bench::fillRandom(hostInput, elementCount, bench::kSeedVectorA);

    double sum = 0.0;
    for (std::size_t i = 0; i < elementCount; ++i) {
        sum += static_cast<double>(hostInput[i]);
    }
    cpuReference = static_cast<float>(sum);
}

// ---------------------------------------------------------------------------
// Fase 3: calentamiento — simétrico al warmUpGpu() de la versión CUDA.
// ---------------------------------------------------------------------------
void warmUpGpu(const bench::ocl::ClEnvironment& env,
               const bench::ocl::ReductionKernel& kernel) {
    constexpr std::size_t warmupCount = 64;

    cl_int status = CL_SUCCESS;
    cl_mem scratchInput = clCreateBuffer(env.context(), CL_MEM_READ_WRITE,
                                         warmupCount * sizeof(float),
                                         nullptr, &status);
    CL_CHECK(status);
    cl_mem scratchPartials = clCreateBuffer(
        env.context(), CL_MEM_READ_WRITE,
        bench::ocl::kReductionStage1Groups * sizeof(float), nullptr, &status);
    CL_CHECK(status);
    cl_mem scratchResult = clCreateBuffer(env.context(), CL_MEM_READ_WRITE,
                                          sizeof(float), nullptr, &status);
    CL_CHECK(status);

    // Mini-reducción completa (ambas etapas), contenido irrelevante.
    cl_event stage1 = nullptr;
    cl_event stage2 = nullptr;
    kernel.enqueue(env.queue(), scratchInput, scratchPartials, scratchResult,
                   warmupCount, &stage1, &stage2);
    CL_CHECK(clWaitForEvents(1, &stage2));  // La etapa 2 implica el fin de la 1.
    CL_CHECK(clReleaseEvent(stage1));
    CL_CHECK(clReleaseEvent(stage2));

    CL_CHECK(clReleaseMemObject(scratchInput));
    CL_CHECK(clReleaseMemObject(scratchPartials));
    CL_CHECK(clReleaseMemObject(scratchResult));
}

// ---------------------------------------------------------------------------
// Fases 4 y 8: gestión de memoria de dispositivo
// ---------------------------------------------------------------------------

/// Reserva entrada (N), parciales (256) y resultado (1) en la GPU.
DeviceBuffers allocateDevice(cl_context context, std::size_t elementCount) {
    DeviceBuffers buffers;
    cl_int status = CL_SUCCESS;

    buffers.input = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   elementCount * sizeof(float), nullptr, &status);
    CL_CHECK(status);
    // Los parciales se escriben en la etapa 1 y se leen en la 2: READ_WRITE.
    buffers.partials = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      bench::ocl::kReductionStage1Groups * sizeof(float),
                                      nullptr, &status);
    CL_CHECK(status);
    buffers.result = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float), nullptr, &status);
    CL_CHECK(status);
    return buffers;
}

/// Libera los buffers de GPU. Se llama SIEMPRE, incluso si la verificación falla.
void freeDevice(DeviceBuffers& buffers) {
    if (buffers.input)    CL_CHECK(clReleaseMemObject(buffers.input));
    if (buffers.partials) CL_CHECK(clReleaseMemObject(buffers.partials));
    if (buffers.result)   CL_CHECK(clReleaseMemObject(buffers.result));
    buffers = DeviceBuffers{};  // Evitar handles colgantes.
}

// ---------------------------------------------------------------------------
// Fase 5: pipeline medido (H2D → kernel → D2H)
// ---------------------------------------------------------------------------

/// Copia el vector de entrada del host a la GPU, midiendo con eventos. Devuelve ms.
double copyHostToDevice(cl_command_queue queue,
                        const std::vector<float>& hostInput,
                        const DeviceBuffers& buffers) {
    const std::size_t bytes = hostInput.size() * sizeof(float);

    cl_event event = nullptr;
    CL_CHECK(clEnqueueWriteBuffer(queue, buffers.input, CL_TRUE, 0, bytes,
                                  hostInput.data(), 0, nullptr, &event));
    const double ms = bench::ocl::eventElapsedMs(event);
    CL_CHECK(clReleaseEvent(event));
    return ms;
}

/// Ejecuta la reducción completa (2 etapas), midiendo su ejecución. Devuelve ms.
double runKernel(cl_command_queue queue,
                 const bench::ocl::ReductionKernel& kernel,
                 const DeviceBuffers& buffers,
                 std::size_t elementCount) {
    cl_event stage1 = nullptr;
    cl_event stage2 = nullptr;
    kernel.enqueue(queue, buffers.input, buffers.partials, buffers.result,
                   elementCount, &stage1, &stage2);

    // Kernel_ms = etapa 1 + etapa 2 (el algoritmo completo).
    const double ms = bench::ocl::eventElapsedMs(stage1) +
                      bench::ocl::eventElapsedMs(stage2);
    CL_CHECK(clReleaseEvent(stage1));
    CL_CHECK(clReleaseEvent(stage2));
    return ms;
}

/// Copia el escalar resultado (4 bytes) de la GPU al host. Devuelve ms.
double copyDeviceToHost(cl_command_queue queue,
                        float& hostResult,
                        const DeviceBuffers& buffers) {
    cl_event event = nullptr;
    CL_CHECK(clEnqueueReadBuffer(queue, buffers.result, CL_TRUE, 0,
                                 sizeof(float), &hostResult,
                                 0, nullptr, &event));
    const double ms = bench::ocl::eventElapsedMs(event);
    CL_CHECK(clReleaseEvent(event));
    return ms;
}

/// Pipeline completo H2D → kernel → D2H; Total_ms con reloj monotónico de host.
Timings runMeasuredPipeline(cl_command_queue queue,
                            const bench::ocl::ReductionKernel& kernel,
                            const std::vector<float>& hostInput,
                            float& hostResult,
                            const DeviceBuffers& buffers,
                            std::size_t elementCount) {
    Timings timings;

    const auto totalStart = std::chrono::steady_clock::now();

    timings.hostToDeviceMs = copyHostToDevice(queue, hostInput, buffers);
    timings.kernelMs       = runKernel(queue, kernel, buffers, elementCount);
    timings.deviceToHostMs = copyDeviceToHost(queue, hostResult, buffers);

    const auto totalEnd = std::chrono::steady_clock::now();
    timings.totalMs =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    return timings;
}

}  // namespace

// ---------------------------------------------------------------------------
// main: solo coordina; misma estructura y orden que cuda/reduction/main.cu.
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // ---- 1. Argumentos de línea de comandos -------------------------------
    const auto config = bench::parseArgs(argc, argv, kDefaultCsvPath);
    if (!config) {
        return EXIT_FAILURE;  // parseArgs ya explicó el problema por stderr.
    }

    // ---- 2. Datos del host (reproducibles) y referencia CPU ---------------
    std::vector<float> hostInput;
    float cpuReference = 0.0f;
    initializeHostData(config->problemSize, hostInput, cpuReference);

    // Resultado escalar, prellenado a cero (misma razón que en CUDA).
    float hostResult = 0.0f;

    // ---- 3. Entorno OpenCL + calentamiento --------------------------------
    bench::ocl::ClEnvironment env;
    bench::ocl::printDeviceInfo(env.device());
    std::cerr << "[INFO] N = " << config->problemSize
              << " | iteracion = " << config->iteration
              << " | csv = " << config->csvPath << '\n';

    bench::ocl::ReductionKernel kernel(env.context(), env.device());
    warmUpGpu(env, kernel);

    // ---- 4. Memoria de dispositivo ----------------------------------------
    DeviceBuffers buffers = allocateDevice(env.context(), config->problemSize);

    // ---- 5. Pipeline medido ------------------------------------------------
    const Timings timings = runMeasuredPipeline(
        env.queue(), kernel, hostInput, hostResult, buffers, config->problemSize);

    // ---- 6. Verificación funcional -----------------------------------------
    // Escalar: se reutiliza verifyElementwise con tamaño 1 (mismo criterio
    // y mismo código compartido que en la versión CUDA).
    const bench::VerificationResult verdict = bench::verifyElementwise(
        &cpuReference, &hostResult, 1, kRelativeTolerance);
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
    freeDevice(buffers);

    // Código de salida útil para los scripts: 0 solo si la verificación pasó.
    return verdict.passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
