/**
 * @file main.cu
 * @brief Benchmark Reduction (CUDA) — orquestador.
 *        MISMAS 8 FASES que cuda/vector_add/main.cu.
 *
 * DIFERENCIAS RESPECTO DE vector_add (semánticas, no estructurales):
 *
 *   - UN solo operando de entrada (el vector A); no existe B.
 *   - El resultado es UN escalar: D2H copia 4 bytes. Eso hace que la
 *     proporción entre cómputo y transferencia sea muy distinta a la de
 *     vector_add — otro punto de comparación para el estudio.
 *   - Kernel_ms cubre las DOS etapas de la reducción (es el algoritmo
 *     completo; ver cuda/reduction/kernel.cu).
 *   - REFERENCIA CPU EN DOUBLE: la suma secuencial de 10⁷ floats acumula un
 *     error de redondeo propio mayor que el de la GPU (que suma en árbol).
 *     Acumular la referencia en double (error despreciable) da un patrón
 *     "verdadero" contra el que comparar AMBOS frameworks con la misma vara.
 *   - Verificación: el escalar de la GPU contra la referencia con tolerancia
 *     relativa 1e-4. Cota del error GPU: cada hilo suma ~N/65536 términos
 *     secuencialmente (~153 con N=10⁷) más ~16 niveles de árbol → error
 *     relativo ≲ 2e-5 con datos positivos. 1e-4 la cubre con margen y
 *     detectaría igualmente cualquier elemento omitido o duplicado (error
 *     relativo ~1/N·suma ≈ 1e-7... pero un bloque entero omitido ≈ 4e-3).
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
constexpr const char* kBenchmarkName  = "Reduction";
constexpr const char* kDefaultCsvPath = "results/cuda/reduction.csv";

/// Tolerancia relativa de verificación (justificación en la cabecera).
constexpr float kRelativeTolerance = 1e-4f;

// ---------------------------------------------------------------------------
// Tipos auxiliares
// ---------------------------------------------------------------------------

/// Buffers en GPU: entrada, parciales de la etapa 1 y resultado escalar.
struct DeviceBuffers {
    float* input    = nullptr;  ///< Vector de entrada (N elementos).
    float* partials = nullptr;  ///< Sumas parciales (una por bloque).
    float* result   = nullptr;  ///< Suma total (1 elemento).
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
 * @brief Genera el vector de entrada (semilla fija) y la referencia CPU.
 *
 * Misma semilla del operando A que en los demás benchmarks: datos
 * reproducibles e idénticos a los de la versión OpenCL.
 *
 * La referencia se acumula en DOUBLE y se redondea a float al final: con
 * 53 bits de mantisa el error del patrón es despreciable (≪ tolerancia),
 * de modo que la verificación mide solo el error de la GPU.
 */
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
// Fase 3: calentamiento de la GPU (mismo criterio que vector_add)
// ---------------------------------------------------------------------------

/**
 * @brief Absorbe creación de contexto y carga del módulo fuera de la medición.
 *
 * Mini-reducción de 64 elementos sin inicializar: el contenido es
 * irrelevante, solo importa que el kernel (sus DOS etapas) se ejecute una
 * vez antes de la zona medida.
 */
void warmUpGpu() {
    CUDA_CHECK(cudaFree(nullptr));  // Fuerza la creación del contexto CUDA.

    constexpr std::size_t warmupCount = 64;
    float* scratchInput    = nullptr;
    float* scratchPartials = nullptr;
    float* scratchResult   = nullptr;
    CUDA_CHECK(cudaMalloc(&scratchInput, warmupCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratchPartials,
                          bench::cuda::kReductionStage1Blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratchResult, sizeof(float)));

    bench::cuda::launchReduction(scratchInput, scratchPartials, scratchResult,
                                 warmupCount);
    CUDA_CHECK(cudaDeviceSynchronize());  // Esperar a que termine de verdad.

    CUDA_CHECK(cudaFree(scratchInput));
    CUDA_CHECK(cudaFree(scratchPartials));
    CUDA_CHECK(cudaFree(scratchResult));
}

// ---------------------------------------------------------------------------
// Fases 4 y 8: gestión de memoria de dispositivo
// ---------------------------------------------------------------------------

/// Reserva entrada (N), parciales (256) y resultado (1) en la GPU.
DeviceBuffers allocateDevice(std::size_t elementCount) {
    DeviceBuffers buffers;
    CUDA_CHECK(cudaMalloc(&buffers.input, elementCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.partials,
                          bench::cuda::kReductionStage1Blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.result, sizeof(float)));
    return buffers;
}

/// Libera los buffers de GPU. Se llama SIEMPRE, incluso si la verificación falla.
void freeDevice(DeviceBuffers& buffers) {
    CUDA_CHECK(cudaFree(buffers.input));
    CUDA_CHECK(cudaFree(buffers.partials));
    CUDA_CHECK(cudaFree(buffers.result));
    buffers = DeviceBuffers{};  // Evitar punteros colgantes.
}

// ---------------------------------------------------------------------------
// Fase 5: pipeline medido (H2D → kernel → D2H)
// ---------------------------------------------------------------------------

/// Copia el vector de entrada del host a la GPU, midiendo con eventos. Devuelve ms.
double copyHostToDevice(const std::vector<float>& hostInput,
                        const DeviceBuffers& buffers) {
    const std::size_t bytes = hostInput.size() * sizeof(float);

    bench::cuda::EventTimer timer;
    timer.start();
    CUDA_CHECK(cudaMemcpy(buffers.input, hostInput.data(), bytes,
                          cudaMemcpyHostToDevice));
    timer.stop();
    return timer.elapsedMs();
}

/// Ejecuta la reducción completa (2 etapas), midiendo su ejecución. Devuelve ms.
double runKernel(const DeviceBuffers& buffers, std::size_t elementCount) {
    bench::cuda::EventTimer timer;
    timer.start();
    // Las dos etapas van al mismo stream entre los dos eventos: Kernel_ms
    // es el tiempo del ALGORITMO completo, comparable con OpenCL.
    bench::cuda::launchReduction(buffers.input, buffers.partials,
                                 buffers.result, elementCount);
    timer.stop();
    return timer.elapsedMs();
}

/// Copia el escalar resultado (4 bytes) de la GPU al host. Devuelve ms.
double copyDeviceToHost(float& hostResult, const DeviceBuffers& buffers) {
    bench::cuda::EventTimer timer;
    timer.start();
    CUDA_CHECK(cudaMemcpy(&hostResult, buffers.result, sizeof(float),
                          cudaMemcpyDeviceToHost));
    timer.stop();
    return timer.elapsedMs();
}

/// Pipeline completo H2D → kernel → D2H; Total_ms con reloj monotónico de host.
Timings runMeasuredPipeline(const std::vector<float>& hostInput,
                            float& hostResult,
                            const DeviceBuffers& buffers,
                            std::size_t elementCount) {
    Timings timings;

    const auto totalStart = std::chrono::steady_clock::now();

    timings.hostToDeviceMs = copyHostToDevice(hostInput, buffers);
    timings.kernelMs       = runKernel(buffers, elementCount);
    timings.deviceToHostMs = copyDeviceToHost(hostResult, buffers);

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

    // Trazabilidad: hardware y parámetros de esta corrida.
    bench::cuda::printDeviceInfo();
    std::cerr << "[INFO] N = " << config->problemSize
              << " | iteracion = " << config->iteration
              << " | csv = " << config->csvPath << '\n';

    // ---- 2. Datos del host (reproducibles) y referencia CPU ---------------
    std::vector<float> hostInput;
    float cpuReference = 0.0f;
    initializeHostData(config->problemSize, hostInput, cpuReference);

    // Resultado escalar, prellenado a cero (si el kernel no escribiera nada,
    // la verificación lo detectaría: 0 != referencia).
    float hostResult = 0.0f;

    // ---- 3. Calentamiento (excluye costes de primera vez) -----------------
    warmUpGpu();

    // ---- 4. Memoria de dispositivo ----------------------------------------
    DeviceBuffers buffers = allocateDevice(config->problemSize);

    // ---- 5. Pipeline medido ------------------------------------------------
    const Timings timings =
        runMeasuredPipeline(hostInput, hostResult, buffers, config->problemSize);

    // ---- 6. Verificación funcional -----------------------------------------
    // El resultado es un escalar: se reutiliza verifyElementwise (DRY) con
    // tamaño 1 — mismo criterio |ref - res| <= tol * max(|ref|, 1).
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
