/**
 * @file main.cpp
 * @brief Benchmark Vector Addition (OpenCL) — orquestador.
 *        ESPEJO 1:1 de cuda/vector_add/main.cu.
 *
 * PIPELINE DE UNA MEDICIÓN (mismas 8 fases que la versión CUDA):
 *
 *   1. parseArgs           →  N, iteración, ruta CSV      [include/common — COMPARTIDO]
 *   2. initializeHostData  →  A, B reproducibles + ref    [include/common — COMPARTIDO]
 *   3. warm-up             →  contexto + COMPILACIÓN ONLINE del kernel
 *   4. allocateDevice      →  clCreateBuffer de A, B, C
 *   5. pipeline medido:
 *        copyHostToDevice  →  HostToDevice_ms   (eventos de profiling)
 *        runKernel         →  Kernel_ms         (eventos de profiling)
 *        copyDeviceToHost  →  DeviceToHost_ms   (eventos de profiling)
 *        (todo envuelto)   →  Total_ms          (std::chrono::steady_clock)
 *   6. verifyResults       →  PASS / FAIL       [include/common — COMPARTIDO]
 *   7. CsvLogger::append   →  fila en el CSV    [include/common — COMPARTIDO]
 *   8. liberación          →  clReleaseMemObject (RAII para contexto/kernel)
 *
 * PUNTO METODOLÓGICO CENTRAL: las fases 1, 2, 6 y 7 usan EXACTAMENTE el
 * mismo código que la versión CUDA (los headers de include/common/). Las
 * dos implementaciones solo difieren en el código GPU — la variable que la
 * investigación aísla. Mismas semillas → mismos datos → misma referencia.
 *
 * NOTA sobre Total_ms: igual que en CUDA, se mide con reloj monotónico del
 * host alrededor de todo el pipeline; el excedente sobre H2D+Kernel+D2H es
 * el overhead de API del framework, uno de los datos comparativos del estudio.
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
// Solo cambian estas dos cadenas y la ruta respecto de la versión CUDA.
// ---------------------------------------------------------------------------
constexpr const char* kFramework      = "OpenCL";
constexpr const char* kBenchmarkName  = "VectorAddition";
constexpr const char* kDefaultCsvPath = "results/opencl/vector_add.csv";

// ---------------------------------------------------------------------------
// Tipos auxiliares
// ---------------------------------------------------------------------------

/// Buffers de los tres vectores en memoria de la GPU (cl_mem ↔ float* CUDA).
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
// Fase 2: inicialización de datos en el host — IDÉNTICA a la versión CUDA.
// Mismas semillas fijas ⇒ los dos frameworks procesan los mismos bytes.
// ---------------------------------------------------------------------------
void initializeHostData(std::size_t elementCount,
                        std::vector<float>& hostA,
                        std::vector<float>& hostB,
                        std::vector<float>& cpuReference) {
    bench::fillRandom(hostA, elementCount, bench::kSeedVectorA);
    bench::fillRandom(hostB, elementCount, bench::kSeedVectorB);

    cpuReference.resize(elementCount);
    for (std::size_t i = 0; i < elementCount; ++i) {
        cpuReference[i] = hostA[i] + hostB[i];  // Referencia secuencial en CPU.
    }
}

// ---------------------------------------------------------------------------
// Fase 3: calentamiento — el análogo simétrico del warmUpGpu() de CUDA.
// ---------------------------------------------------------------------------

/**
 * @brief Absorbe los costes de "primera vez" de OpenCL fuera de la medición.
 *
 * En CUDA el warm-up absorbía: creación de contexto + carga del módulo.
 * En OpenCL absorbe sus equivalentes exactos:
 *   - creación de plataforma/contexto/cola (ya hecha al construir env),
 *   - COMPILACIÓN ONLINE del kernel (clBuildProgram, al construir kernel),
 *   - primer encolado real (activación de la cola y el dispositivo).
 *
 * SIMETRÍA METODOLÓGICA: ambos frameworks se miden en régimen estacionario;
 * ninguno paga sus costes de arranque dentro de la zona medida.
 */
void warmUpGpu(const bench::ocl::ClEnvironment& env,
               const bench::ocl::VectorAddKernel& kernel) {
    constexpr std::size_t warmupCount = 64;
    constexpr std::size_t warmupBytes = warmupCount * sizeof(float);

    cl_int status = CL_SUCCESS;
    cl_mem scratch = clCreateBuffer(env.context(), CL_MEM_READ_WRITE,
                                    warmupBytes, nullptr, &status);
    CL_CHECK(status);

    // Mini-lanzamiento (contenido irrelevante: solo activa el pipeline).
    cl_event event = nullptr;
    kernel.enqueue(env.queue(), scratch, scratch, scratch, warmupCount, &event);
    CL_CHECK(clWaitForEvents(1, &event));   // Esperar a que termine de verdad.
    CL_CHECK(clReleaseEvent(event));
    CL_CHECK(clReleaseMemObject(scratch));
}

// ---------------------------------------------------------------------------
// Fases 4 y 8: gestión de memoria de dispositivo
// ---------------------------------------------------------------------------

/**
 * @brief Reserva los tres buffers en la GPU — espejo de los cudaMalloc.
 *
 * Los flags reflejan el uso real (A y B solo lectura por el kernel, C solo
 * escritura): mejor documentación y el driver puede optimizar.
 */
DeviceBuffers allocateDevice(cl_context context, std::size_t elementCount) {
    const std::size_t bytes = elementCount * sizeof(float);
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
// Fase 5: pipeline medido (H2D → kernel → D2H)
// ---------------------------------------------------------------------------

/**
 * @brief Copia A y B del host a la GPU, midiendo con eventos. Devuelve ms.
 *
 * clEnqueueWriteBuffer con CL_TRUE (bloqueante) es el análogo directo del
 * cudaMemcpy síncrono de la versión CUDA. El tiempo reportado NO es el de
 * la llamada bloqueante, sino el intervalo COMMAND_START→END de cada
 * transferencia según el reloj de la GPU — igual que los eventos CUDA.
 * Se suman las dos transferencias: mismo alcance que el EventTimer CUDA,
 * que también envolvía ambas copias.
 */
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
                 const bench::ocl::VectorAddKernel& kernel,
                 const DeviceBuffers& buffers,
                 std::size_t elementCount) {
    cl_event event = nullptr;
    kernel.enqueue(queue, buffers.a, buffers.b, buffers.c, elementCount, &event);
    // eventElapsedMs espera al evento: cuando retorna, el kernel terminó de
    // verdad (profiling mide ejecución en GPU, no el encolado) — igual que
    // el elapsedMs() del EventTimer CUDA.
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

/**
 * @brief Ejecuta el pipeline completo H2D → kernel → D2H y recoge los 4 tiempos.
 *
 * Total_ms: mismo criterio que en CUDA — std::chrono::steady_clock
 * (monotónico) alrededor de TODO el pipeline, incluido el overhead de API
 * que los eventos de profiling no ven.
 */
Timings runMeasuredPipeline(cl_command_queue queue,
                            const bench::ocl::VectorAddKernel& kernel,
                            const std::vector<float>& hostA,
                            const std::vector<float>& hostB,
                            std::vector<float>& hostC,
                            const DeviceBuffers& buffers,
                            std::size_t elementCount) {
    Timings timings;

    const auto totalStart = std::chrono::steady_clock::now();

    timings.hostToDeviceMs = copyHostToDevice(queue, hostA, hostB, buffers);
    timings.kernelMs       = runKernel(queue, kernel, buffers, elementCount);
    timings.deviceToHostMs = copyDeviceToHost(queue, hostC, buffers);

    const auto totalEnd = std::chrono::steady_clock::now();
    timings.totalMs =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    return timings;
}

}  // namespace

// ---------------------------------------------------------------------------
// main: solo coordina; toda la lógica vive en las funciones anteriores.
// Misma estructura, mismo orden y mismos comentarios que main.cu.
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

    // Buffer de resultados, prellenado a cero (misma razón que en CUDA).
    std::vector<float> hostC(config->problemSize, 0.0f);

    // ---- 3. Entorno OpenCL + calentamiento --------------------------------
    // Construir env y kernel YA ES parte del warm-up: contexto, cola y
    // compilación online del kernel ocurren aquí, fuera de la zona medida.
    bench::ocl::ClEnvironment env;
    bench::ocl::printDeviceInfo(env.device());
    std::cerr << "[INFO] N = " << config->problemSize
              << " | iteracion = " << config->iteration
              << " | csv = " << config->csvPath << '\n';

    bench::ocl::VectorAddKernel kernel(env.context(), env.device());
    warmUpGpu(env, kernel);

    // ---- 4. Memoria de dispositivo ----------------------------------------
    DeviceBuffers buffers = allocateDevice(env.context(), config->problemSize);

    // ---- 5. Pipeline medido ------------------------------------------------
    const Timings timings = runMeasuredPipeline(
        env.queue(), kernel, hostA, hostB, hostC, buffers, config->problemSize);

    // ---- 6. Verificación funcional (100% de los elementos) -----------------
    const bench::VerificationResult verdict = bench::verifyElementwise(
        cpuReference.data(), hostC.data(), config->problemSize);
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
