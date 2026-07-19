/**
 * @file cuda_utils.cuh
 * @brief Utilidades comunes para todos los benchmarks CUDA del proyecto.
 *
 * Contiene únicamente infraestructura ESPECÍFICA de CUDA:
 *   - CUDA_CHECK : verificación obligatoria de toda llamada al runtime.
 *   - EventTimer : temporizador RAII basado en CUDA Events.
 *   - printDeviceInfo : registro del hardware para trazabilidad experimental.
 *
 * La infraestructura agnóstica al framework (CSV, argumentos, datos,
 * verificación) vive en include/common/ y NO debe duplicarse aquí.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

/**
 * @brief Verifica el resultado de una llamada al runtime de CUDA.
 *
 * DECISIÓN DE DISEÑO: en un benchmark científico, un error CUDA silencioso
 * es peor que un crash: produciría tiempos y resultados sin sentido que
 * podrían colarse en el artículo. Por eso la política es "fail fast":
 * ante cualquier error se informa archivo, línea, llamada y descripción,
 * y se aborta con EXIT_FAILURE (visible para el script de automatización).
 *
 * Se implementa como macro (y no como función) para poder capturar
 * __FILE__ y __LINE__ del punto de llamada, y el texto de la expresión.
 * El do/while(0) hace que la macro se comporte como una sentencia normal
 * (segura dentro de if/else sin llaves).
 */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t cudaCheckError_ = (call);                           \
        if (cudaCheckError_ != cudaSuccess) {                                 \
            std::fprintf(stderr,                                              \
                         "[CUDA ERROR] %s:%d\n  Llamada : %s\n  Error   : %s\n", \
                         __FILE__, __LINE__, #call,                           \
                         cudaGetErrorString(cudaCheckError_));                \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

namespace bench {
namespace cuda {

/**
 * @brief Temporizador de precisión basado en CUDA Events (patrón RAII).
 *
 * POR QUÉ CUDA EVENTS Y NO RELOJES DE CPU:
 * Las operaciones CUDA se ENCOLAN en un stream y se ejecutan de forma
 * asíncrona respecto a la CPU. Un reloj de host (std::chrono) alrededor de
 * un lanzamiento de kernel mediría solo el coste de encolado (~µs), no la
 * ejecución real. Los CUDA Events se insertan en el propio stream y la GPU
 * les pone marca de tiempo con su reloj hardware (resolución ~0.5 µs),
 * midiendo exactamente el intervalo entre ambos eventos EN LA GPU.
 *
 * El mismo mecanismo es válido para las transferencias cudaMemcpy, que
 * también son operaciones del stream: por eso este temporizador se usa
 * para H2D, kernel y D2H, garantizando que los tres tiempos provienen de
 * la MISMA fuente de reloj (comparables entre sí y con OpenCL, cuyo
 * perfilado de eventos es análogo).
 *
 * RAII: los eventos se crean en el constructor y se destruyen en el
 * destructor, de modo que no pueden filtrarse aunque el flujo de control
 * salga por una ruta inesperada.
 *
 * Uso:
 *     EventTimer timer;
 *     timer.start();
 *     ...operacion CUDA...
 *     timer.stop();
 *     float ms = timer.elapsedMs();   // sincroniza y devuelve milisegundos
 */
class EventTimer {
public:
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&startEvent_));
        CUDA_CHECK(cudaEventCreate(&stopEvent_));
    }

    ~EventTimer() {
        // En el destructor no se usa CUDA_CHECK: destruir nunca debe abortar.
        cudaEventDestroy(startEvent_);
        cudaEventDestroy(stopEvent_);
    }

    // Un temporizador posee recursos GPU únicos: prohibimos copiarlo para
    // evitar dobles liberaciones (regla de SOLID/RAII sobre propiedad única).
    EventTimer(const EventTimer&)            = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    /// Inserta el evento de inicio en el stream indicado (0 = stream por defecto).
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(startEvent_, stream));
    }

    /// Inserta el evento de fin en el stream indicado.
    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stopEvent_, stream));
    }

    /**
     * @brief Devuelve el tiempo transcurrido entre start() y stop(), en ms.
     *
     * cudaEventSynchronize bloquea la CPU hasta que la GPU haya alcanzado el
     * evento de fin: sin esta sincronización se leería un tiempo inválido.
     */
    float elapsedMs() {
        CUDA_CHECK(cudaEventSynchronize(stopEvent_));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent_, stopEvent_));
        return milliseconds;
    }

private:
    cudaEvent_t startEvent_{};  ///< Marca de tiempo de inicio (reloj GPU).
    cudaEvent_t stopEvent_{};   ///< Marca de tiempo de fin (reloj GPU).
};

/**
 * @brief Imprime en stderr las características del dispositivo activo.
 *
 * TRAZABILIDAD: registrar el hardware junto a cada sesión experimental
 * permite detectar a posteriori si alguna medición se ejecutó en un equipo
 * distinto al declarado en el artículo. Va a stderr para no interferir con
 * la salida funcional (PASS/FAIL) de stdout.
 */
inline void printDeviceInfo(int deviceId = 0) {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, deviceId));

    std::fprintf(stderr,
                 "[INFO] GPU: %s | CC %d.%d | %zu MB | %d SMs | Driver CUDA %d.%d\n",
                 properties.name,
                 properties.major, properties.minor,
                 properties.totalGlobalMem / (1024 * 1024),
                 properties.multiProcessorCount,
                 CUDART_VERSION / 1000, (CUDART_VERSION % 100) / 10);
}

}  // namespace cuda
}  // namespace bench
