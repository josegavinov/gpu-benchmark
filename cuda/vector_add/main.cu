/**
 * @file main.cu
 * @brief Benchmark Vector Addition (CUDA) — orquestador.
 *
 * PIPELINE DE UNA MEDICIÓN (cada fase es una función separada):
 *
 *   1. parseArgs           →  N, iteración, ruta CSV (línea de comandos)
 *   2. initializeHostData  →  A, B reproducibles + referencia CPU
 *   3. warmUpGpu           →  excluir costes de primera vez (contexto CUDA)
 *   4. allocateDevice      →  cudaMalloc de A, B, C en la GPU
 *   5. pipeline medido:
 *        copyHostToDevice  →  HostToDevice_ms   (CUDA Events)
 *        runKernel         →  Kernel_ms         (CUDA Events)
 *        copyDeviceToHost  →  DeviceToHost_ms   (CUDA Events)
 *        (todo envuelto)   →  Total_ms          (std::chrono::steady_clock)
 *   6. verifyResults       →  PASS / FAIL comparando el 100% con la CPU
 *   7. CsvLogger::append   →  fila en el CSV (crea el archivo si no existe)
 *   8. freeDevice          →  liberación de memoria GPU
 *
 * NOTA METODOLÓGICA — Total_ms:
 * Total_ms se mide con un reloj monotónico del HOST alrededor de todo el
 * pipeline GPU. Será ligeramente MAYOR que la suma H2D+Kernel+D2H: la
 * diferencia es el overhead del framework (lanzamientos, sincronizaciones,
 * llamadas al driver). Ese overhead es precisamente una de las cosas que
 * distinguen a CUDA de OpenCL, así que medirlo por separado de los tiempos
 * "puros" de GPU es información valiosa para el artículo.
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

#include "cuda/common/cuda_utils.cuh"

#include "kernel.cuh"

namespace {

// ---------------------------------------------------------------------------
// Constantes de identificación del experimento (columnas fijas del CSV).
// ---------------------------------------------------------------------------
constexpr const char* kFramework      = "CUDA";
constexpr const char* kBenchmarkName  = "VectorAddition";
constexpr const char* kDefaultCsvPath = "results/cuda/vector_add.csv";

// ---------------------------------------------------------------------------
// Tipos auxiliares
// ---------------------------------------------------------------------------

/// Punteros a los tres vectores en memoria de la GPU.
struct DeviceBuffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
};

/// Los cuatro tiempos que exige el protocolo, en milisegundos.
struct Timings {
    double hostToDeviceMs = 0.0;
    double kernelMs       = 0.0;
    double deviceToHostMs = 0.0;
    double totalMs        = 0.0;
};

// ---------------------------------------------------------------------------
// Fase 2: inicialización de datos en el host
// ---------------------------------------------------------------------------

/**
 * @brief Genera A y B (semillas fijas) y calcula la referencia CPU.
 *
 * La referencia C_ref = A + B se calcula ANTES de tocar la GPU: así la
 * verificación posterior es contra un resultado obtenido por una ruta de
 * cómputo totalmente independiente.
 */
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
// Fase 3: calentamiento de la GPU
// ---------------------------------------------------------------------------

/**
 * @brief Absorbe los costes de "primera vez" para que NO contaminen la medición.
 *
 * La primera llamada CUDA de un proceso crea el contexto (decenas o cientos
 * de ms) y el primer lanzamiento carga el módulo del kernel. Si no se
 * aislaran, la iteración 1 de cada corrida mediría "creación de contexto"
 * en lugar de "transferencia H2D", sesgando la estadística.
 *
 * DECISIÓN METODOLÓGICA: el estudio compara rendimiento en RÉGIMEN
 * ESTACIONARIO. En la fase OpenCL haremos el calentamiento análogo
 * (creación de contexto, compilación del kernel), manteniendo la simetría.
 */
void warmUpGpu() {
    // Fuerza la creación del contexto CUDA sin efecto secundario alguno.
    CUDA_CHECK(cudaFree(nullptr));

    // Mini-lanzamiento para forzar la carga del módulo del kernel en el
    // driver. Los datos (64 floats sin inicializar) son irrelevantes:
    // solo importa que el kernel se ejecute una vez.
    constexpr std::size_t warmupCount = 64;
    float* scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, warmupCount * sizeof(float)));
    bench::cuda::launchVectorAdd(scratch, scratch, scratch, warmupCount);
    CUDA_CHECK(cudaDeviceSynchronize());  // Esperar a que termine de verdad.
    CUDA_CHECK(cudaFree(scratch));
}

// ---------------------------------------------------------------------------
// Fases 4 y 8: gestión de memoria de dispositivo
// ---------------------------------------------------------------------------

/// Reserva los tres buffers en la GPU (aborta con diagnóstico si no hay memoria).
DeviceBuffers allocateDevice(std::size_t elementCount) {
    const std::size_t bytes = elementCount * sizeof(float);
    DeviceBuffers buffers;
    CUDA_CHECK(cudaMalloc(&buffers.a, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.b, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.c, bytes));
    return buffers;
}

/// Libera los buffers de GPU. Se llama SIEMPRE, incluso si la verificación falla.
void freeDevice(DeviceBuffers& buffers) {
    // cudaFree(nullptr) es inofensivo, por lo que no hace falta comprobar.
    CUDA_CHECK(cudaFree(buffers.a));
    CUDA_CHECK(cudaFree(buffers.b));
    CUDA_CHECK(cudaFree(buffers.c));
    buffers = DeviceBuffers{};  // Evitar punteros colgantes.
}

// ---------------------------------------------------------------------------
// Fase 5: pipeline medido (H2D → kernel → D2H)
// ---------------------------------------------------------------------------

/// Copia A y B del host a la GPU, midiendo con eventos CUDA. Devuelve ms.
double copyHostToDevice(const std::vector<float>& hostA,
                        const std::vector<float>& hostB,
                        const DeviceBuffers& buffers) {
    const std::size_t bytes = hostA.size() * sizeof(float);

    bench::cuda::EventTimer timer;
    timer.start();
    // cudaMemcpyHostToDevice: copia por el bus PCIe hacia la memoria GDDR5.
    // Con memoria host paginable esta llamada es síncrona, pero al medir con
    // eventos delimitamos exactamente el intervalo ocupado en el stream.
    CUDA_CHECK(cudaMemcpy(buffers.a, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.b, hostB.data(), bytes, cudaMemcpyHostToDevice));
    timer.stop();
    return timer.elapsedMs();
}

/// Ejecuta el kernel una vez, midiendo SOLO su ejecución. Devuelve ms.
double runKernel(const DeviceBuffers& buffers, std::size_t elementCount) {
    bench::cuda::EventTimer timer;
    timer.start();
    bench::cuda::launchVectorAdd(buffers.a, buffers.b, buffers.c, elementCount);
    timer.stop();
    // elapsedMs() sincroniza con el evento de fin: cuando retorna, el kernel
    // ha terminado de verdad (los eventos miden GPU, no el encolado).
    return timer.elapsedMs();
}

/// Copia el resultado C de la GPU al host, midiendo con eventos. Devuelve ms.
double copyDeviceToHost(std::vector<float>& hostC, const DeviceBuffers& buffers) {
    const std::size_t bytes = hostC.size() * sizeof(float);

    bench::cuda::EventTimer timer;
    timer.start();
    CUDA_CHECK(cudaMemcpy(hostC.data(), buffers.c, bytes, cudaMemcpyDeviceToHost));
    timer.stop();
    return timer.elapsedMs();
}

/**
 * @brief Ejecuta el pipeline completo H2D → kernel → D2H y recoge los 4 tiempos.
 *
 * Total_ms se mide con std::chrono::steady_clock (monotónico: inmune a
 * ajustes del reloj del sistema, a diferencia de system_clock) alrededor de
 * TODO el pipeline, incluido el overhead de API que los eventos no ven.
 */
Timings runMeasuredPipeline(const std::vector<float>& hostA,
                            const std::vector<float>& hostB,
                            std::vector<float>& hostC,
                            const DeviceBuffers& buffers,
                            std::size_t elementCount) {
    Timings timings;

    const auto totalStart = std::chrono::steady_clock::now();

    timings.hostToDeviceMs = copyHostToDevice(hostA, hostB, buffers);
    timings.kernelMs       = runKernel(buffers, elementCount);
    timings.deviceToHostMs = copyDeviceToHost(hostC, buffers);

    const auto totalEnd = std::chrono::steady_clock::now();
    timings.totalMs =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    return timings;
}

}  // namespace

// ---------------------------------------------------------------------------
// main: solo coordina; toda la lógica vive en las funciones anteriores.
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // ---- 1. Argumentos de línea de comandos -------------------------------
    const auto config = bench::parseArgs(argc, argv, kDefaultCsvPath);
    if (!config) {
        return EXIT_FAILURE;  // parseArgs ya explicó el problema por stderr.
    }

    // Trazabilidad: dejar constancia del hardware usado en esta corrida.
    bench::cuda::printDeviceInfo();
    std::cerr << "[INFO] N = " << config->problemSize
              << " | iteracion = " << config->iteration
              << " | csv = " << config->csvPath << '\n';

    // ---- 2. Datos del host (reproducibles) y referencia CPU ---------------
    std::vector<float> hostA;
    std::vector<float> hostB;
    std::vector<float> cpuReference;
    initializeHostData(config->problemSize, hostA, hostB, cpuReference);

    // Buffer de resultados, prellenado a cero: si el kernel no escribiera
    // nada, la verificación lo detectaría (cero != referencia).
    std::vector<float> hostC(config->problemSize, 0.0f);

    // ---- 3. Calentamiento (excluye costes de primera vez) -----------------
    warmUpGpu();

    // ---- 4. Memoria de dispositivo ----------------------------------------
    DeviceBuffers buffers = allocateDevice(config->problemSize);

    // ---- 5. Pipeline medido ------------------------------------------------
    const Timings timings =
        runMeasuredPipeline(hostA, hostB, hostC, buffers, config->problemSize);

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
    freeDevice(buffers);

    // Código de salida útil para los scripts: 0 solo si la verificación pasó.
    return verdict.passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
