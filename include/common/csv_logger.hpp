/**
 * @file csv_logger.hpp
 * @brief Registro automático de resultados experimentales en formato CSV.
 *
 * REQUISITO EXPERIMENTAL: toda medición debe persistirse automáticamente en
 * un CSV con un esquema fijo, idéntico para CUDA y OpenCL, de modo que el
 * análisis estadístico posterior (Python/R) pueda concatenar y comparar
 * resultados de ambos frameworks sin transformaciones.
 *
 * Esquema (orden y nombres EXACTOS de columnas):
 *
 *   Framework,Benchmark,ProblemSize,Iteration,
 *   HostToDevice_ms,Kernel_ms,DeviceToHost_ms,Total_ms,Status
 *
 * Comportamiento:
 *   - Si el archivo no existe (o está vacío), se crea y se escribe la
 *     cabecera antes de la primera fila.
 *   - Si ya existe, se añade una nueva fila al final (modo append).
 *   - Los directorios intermedios de la ruta se crean automáticamente,
 *     para que los scripts no tengan que preparar el filesystem.
 *
 * Este módulo es C++ puro (sin CUDA/OpenCL): se comparte entre frameworks.
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bench {

/**
 * @brief Resultado completo de una ejecución del benchmark.
 *
 * Corresponde 1:1 con una fila del CSV. Mantenerlo como struct plano
 * desacopla la medición (quién lo rellena) de la persistencia (CsvLogger),
 * siguiendo el principio de responsabilidad única (SRP de SOLID).
 */
struct BenchmarkResult {
    std::string framework;       ///< "CUDA" u "OpenCL".
    std::string benchmark;       ///< P. ej. "VectorAddition".
    std::size_t problemSize = 0; ///< Número de elementos (N).
    int         iteration   = 1; ///< Índice de repetición.
    double hostToDeviceMs = 0.0; ///< Tiempo de copia Host -> Device (ms).
    double kernelMs       = 0.0; ///< Tiempo de ejecución del kernel (ms).
    double deviceToHostMs = 0.0; ///< Tiempo de copia Device -> Host (ms).
    double totalMs        = 0.0; ///< Tiempo total del pipeline GPU (ms).
    std::string status;          ///< "PASS" o "FAIL" tras la verificación.
};

/**
 * @brief Escritor de resultados en CSV con creación/append automáticos.
 *
 * Uso:
 *     bench::CsvLogger logger("results/cuda/vector_add.csv");
 *     logger.append(result);
 */
class CsvLogger {
public:
    /// Cabecera exigida por el protocolo experimental (no modificar).
    static constexpr const char* kHeader =
        "Framework,Benchmark,ProblemSize,Iteration,"
        "HostToDevice_ms,Kernel_ms,DeviceToHost_ms,Total_ms,Status";

    explicit CsvLogger(std::string filePath) : filePath_(std::move(filePath)) {}

    /**
     * @brief Añade una fila al CSV (creando archivo y cabecera si hace falta).
     * @throws std::runtime_error si el archivo no puede abrirse/escribirse.
     */
    void append(const BenchmarkResult& r) const {
        namespace fs = std::filesystem;

        // 1. Garantizar que el directorio destino existe (p. ej. results/cuda/).
        const fs::path path(filePath_);
        if (path.has_parent_path()) {
            fs::create_directories(path.parent_path());
        }

        // 2. Decidir si hay que escribir la cabecera: solo cuando el archivo
        //    no existe todavía o existe pero está vacío.
        const bool needsHeader = !fs::exists(path) || fs::file_size(path) == 0;

        // 3. Abrir en modo append: nunca se sobreescriben mediciones previas.
        std::ofstream file(filePath_, std::ios::out | std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("No se pudo abrir el CSV: " + filePath_);
        }

        if (needsHeader) {
            file << kHeader << '\n';
        }
        file << toRow(r) << '\n';

        // 4. Comprobar el estado del stream: un fallo de escritura silencioso
        //    invalidaría el experimento sin que nadie lo note.
        if (!file.good()) {
            throw std::runtime_error("Fallo al escribir en el CSV: " + filePath_);
        }
    }

private:
    /**
     * @brief Serializa un resultado como fila CSV.
     *
     * Los tiempos se escriben con 6 decimales fijos: en la Quadro K2200 los
     * kernels pequeños duran microsegundos, y con menos precisión se
     * perdería resolución en el análisis estadístico.
     */
    static std::string toRow(const BenchmarkResult& r) {
        std::ostringstream row;
        row << r.framework << ','
            << r.benchmark << ','
            << r.problemSize << ','
            << r.iteration << ','
            << std::fixed << std::setprecision(6)
            << r.hostToDeviceMs << ','
            << r.kernelMs << ','
            << r.deviceToHostMs << ','
            << r.totalMs << ','
            << r.status;
        return row.str();
    }

    std::string filePath_;  ///< Ruta del archivo CSV de destino.
};

}  // namespace bench
