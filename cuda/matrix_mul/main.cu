/**
 * @file main.cu
 * @brief Benchmark Matrix Multiplication (CUDA) — orquestador.
 *        MISMAS 8 FASES que cuda/vector_add/main.cu.
 *
 * DIFERENCIAS RESPECTO DE vector_add (todas de semántica, no de estructura):
 *
 *   - ProblemSize es la DIMENSIÓN N de las matrices cuadradas: los buffers
 *     tienen N² elementos y el kernel hace 2N³ operaciones. El barrido usa
 *     por tanto tamaños mucho menores (p. ej. 128..2048, no 100000..10⁷).
 *   - La referencia CPU es el triple bucle clásico O(N³), en orden i,k,j:
 *     recorre B por filas (amigable con la caché) y acumula sobre k en el
 *     MISMO orden ascendente que el kernel — minimiza la divergencia de
 *     redondeo entre ambas rutas de cómputo.
 *   - Verificación con tolerancia relativa 1e-3 (no la 1e-5 por defecto):
 *     la GPU contrae a*b+c en FMA (un redondeo) y la CPU no; con N=2048
 *     términos positivos la cota del error acumulado es ~N·ε ≈ 2.4e-4.
 *     1e-3 queda por encima de esa cota pero órdenes de magnitud por debajo
 *     de cualquier error real de indexación (que alteraría el valor ~100%).
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
constexpr const char* kBenchmarkName  = "MatrixMultiplication";
constexpr const char* kDefaultCsvPath = "results/cuda/matrix_mul.csv";

/// Tolerancia relativa de verificación (justificación en la cabecera).
constexpr float kRelativeTolerance = 1e-3f;

// ---------------------------------------------------------------------------
// Tipos auxiliares (idénticos a vector_add)
// ---------------------------------------------------------------------------

/// Punteros a las tres matrices en memoria de la GPU.
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
 * @brief Genera A y B (N² elementos, semillas fijas) y la referencia CPU.
 *
 * Mismas semillas del protocolo que vector_add (una por operando): los
 * datos son reproducibles e idénticos a los de la versión OpenCL.
 *
 * Referencia en orden i,k,j: para cada fila i de C se acumula la
 * contribución de A[i][k]·(fila k de B). Ventajas frente al orden i,j,k:
 *   1. B se recorre por filas (localidad de caché ⇒ referencia ~10× más
 *      rápida, importante porque se recalcula en cada uno de los 30 procesos).
 *   2. La acumulación sobre k es ascendente, igual que en el kernel tiled.
 */
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
// Fase 3: calentamiento de la GPU (mismo criterio que vector_add)
// ---------------------------------------------------------------------------

/**
 * @brief Absorbe creación de contexto y carga del módulo fuera de la medición.
 *
 * Mini-multiplicación de 64×64 con datos sin inicializar: el contenido es
 * irrelevante, solo importa que ESTE kernel se ejecute una vez para que su
 * módulo quede cargado en el driver antes de la zona medida.
 */
void warmUpGpu() {
    CUDA_CHECK(cudaFree(nullptr));  // Fuerza la creación del contexto CUDA.

    constexpr std::size_t warmupDim   = 64;
    constexpr std::size_t warmupBytes = warmupDim * warmupDim * sizeof(float);
    float* scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, warmupBytes));
    bench::cuda::launchMatrixMul(scratch, scratch, scratch, warmupDim);
    CUDA_CHECK(cudaDeviceSynchronize());  // Esperar a que termine de verdad.
    CUDA_CHECK(cudaFree(scratch));
}

// ---------------------------------------------------------------------------
// Fases 4 y 8: gestión de memoria de dispositivo
// ---------------------------------------------------------------------------

/// Reserva las tres matrices N×N en la GPU.
DeviceBuffers allocateDevice(std::size_t dimension) {
    const std::size_t bytes = dimension * dimension * sizeof(float);
    DeviceBuffers buffers;
    CUDA_CHECK(cudaMalloc(&buffers.a, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.b, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.c, bytes));
    return buffers;
}

/// Libera los buffers de GPU. Se llama SIEMPRE, incluso si la verificación falla.
void freeDevice(DeviceBuffers& buffers) {
    CUDA_CHECK(cudaFree(buffers.a));
    CUDA_CHECK(cudaFree(buffers.b));
    CUDA_CHECK(cudaFree(buffers.c));
    buffers = DeviceBuffers{};  // Evitar punteros colgantes.
}

// ---------------------------------------------------------------------------
// Fase 5: pipeline medido (H2D → kernel → D2H) — mismo patrón que vector_add
// ---------------------------------------------------------------------------

/// Copia A y B del host a la GPU, midiendo con eventos CUDA. Devuelve ms.
double copyHostToDevice(const std::vector<float>& hostA,
                        const std::vector<float>& hostB,
                        const DeviceBuffers& buffers) {
    const std::size_t bytes = hostA.size() * sizeof(float);

    bench::cuda::EventTimer timer;
    timer.start();
    CUDA_CHECK(cudaMemcpy(buffers.a, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.b, hostB.data(), bytes, cudaMemcpyHostToDevice));
    timer.stop();
    return timer.elapsedMs();
}

/// Ejecuta el kernel una vez, midiendo SOLO su ejecución. Devuelve ms.
double runKernel(const DeviceBuffers& buffers, std::size_t dimension) {
    bench::cuda::EventTimer timer;
    timer.start();
    bench::cuda::launchMatrixMul(buffers.a, buffers.b, buffers.c, dimension);
    timer.stop();
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

/// Pipeline completo H2D → kernel → D2H; Total_ms con reloj monotónico de host.
Timings runMeasuredPipeline(const std::vector<float>& hostA,
                            const std::vector<float>& hostB,
                            std::vector<float>& hostC,
                            const DeviceBuffers& buffers,
                            std::size_t dimension) {
    Timings timings;

    const auto totalStart = std::chrono::steady_clock::now();

    timings.hostToDeviceMs = copyHostToDevice(hostA, hostB, buffers);
    timings.kernelMs       = runKernel(buffers, dimension);
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

    // Trazabilidad: hardware y parámetros de esta corrida.
    bench::cuda::printDeviceInfo();
    std::cerr << "[INFO] N = " << config->problemSize << " (matriz "
              << config->problemSize << "x" << config->problemSize << ")"
              << " | iteracion = " << config->iteration
              << " | csv = " << config->csvPath << '\n';

    // ---- 2. Datos del host (reproducibles) y referencia CPU ---------------
    std::vector<float> hostA;
    std::vector<float> hostB;
    std::vector<float> cpuReference;
    initializeHostData(config->problemSize, hostA, hostB, cpuReference);

    // Buffer de resultados prellenado a cero (misma razón que en vector_add).
    std::vector<float> hostC(config->problemSize * config->problemSize, 0.0f);

    // ---- 3. Calentamiento (excluye costes de primera vez) -----------------
    warmUpGpu();

    // ---- 4. Memoria de dispositivo ----------------------------------------
    DeviceBuffers buffers = allocateDevice(config->problemSize);

    // ---- 5. Pipeline medido ------------------------------------------------
    const Timings timings =
        runMeasuredPipeline(hostA, hostB, hostC, buffers, config->problemSize);

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
    freeDevice(buffers);

    // Código de salida útil para los scripts: 0 solo si la verificación pasó.
    return verdict.passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
