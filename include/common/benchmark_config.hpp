/**
 * @file benchmark_config.hpp
 * @brief Configuración del benchmark a partir de la línea de comandos.
 *
 * DECISIÓN DE DISEÑO:
 * Cada ejecución del binario realiza UNA sola medición. Las repeticiones
 * (p. ej. 30 iteraciones) las controla un script Bash externo, que invoca
 * el programa muchas veces. Esto tiene dos ventajas científicas:
 *
 *   1. Cada iteración es un proceso independiente: no hay estado residual
 *      (cachés de asignación, contexto CUDA "caliente") que contamine las
 *      mediciones entre repeticiones.
 *   2. El mismo binario sirve para exploración manual y para automatización
 *      masiva sin recompilar ni modificar código.
 *
 * Interfaz de línea de comandos:
 *
 *     ./vector_add <tamano_problema> [iteracion] [ruta_csv]
 *
 *   - tamano_problema : número de elementos del vector (obligatorio).
 *   - iteracion       : número de repetición que asigna el script externo
 *                       (opcional, por defecto 1). Se vuelca tal cual a la
 *                       columna "Iteration" del CSV.
 *   - ruta_csv        : archivo CSV de salida (opcional; cada benchmark
 *                       define su ruta por defecto, p. ej.
 *                       results/cuda/vector_add.csv).
 *
 * Este módulo es C++ puro: NO depende de CUDA ni de OpenCL, por lo que se
 * reutiliza sin cambios en ambos frameworks (principio DRY).
 */

#pragma once

#include <cstddef>
#include <iostream>
#include <optional>
#include <string>

namespace bench {

/**
 * @brief Parámetros de una ejecución individual del benchmark.
 */
struct BenchmarkConfig {
    std::size_t problemSize = 0;  ///< Número de elementos del problema (N).
    int         iteration   = 1;  ///< Índice de repetición (lo asigna el script).
    std::string csvPath;          ///< Ruta del CSV donde se registra el resultado.
};

/**
 * @brief Imprime en stderr el modo de uso del programa.
 *
 * Se imprime en stderr (no en stdout) para no contaminar posibles
 * redirecciones de la salida estándar en los scripts de automatización.
 */
inline void printUsage(const char* programName, const std::string& defaultCsvPath) {
    std::cerr
        << "Uso: " << programName << " <tamano_problema> [iteracion] [ruta_csv]\n"
        << "\n"
        << "  tamano_problema  Numero de elementos del vector (entero > 0).\n"
        << "  iteracion        Numero de repeticion para el CSV (por defecto: 1).\n"
        << "  ruta_csv         Archivo CSV de salida (por defecto: "
        << defaultCsvPath << ").\n"
        << "\n"
        << "Ejemplos:\n"
        << "  " << programName << " 100000\n"
        << "  " << programName << " 5000000 12 results/cuda/vector_add.csv\n";
}

/**
 * @brief Analiza y valida los argumentos de línea de comandos.
 *
 * @param argc            Contador de argumentos de main().
 * @param argv            Vector de argumentos de main().
 * @param defaultCsvPath  Ruta CSV usada si el usuario no indica una.
 * @return La configuración validada, o std::nullopt si los argumentos son
 *         inválidos (en cuyo caso ya se imprimió el mensaje de uso).
 *
 * DECISIÓN DE DISEÑO: se devuelve std::optional en lugar de terminar el
 * proceso aquí dentro. Así, la política de salida (código de retorno) queda
 * en main(), y esta función es pura y testeable.
 */
inline std::optional<BenchmarkConfig> parseArgs(int argc, char** argv,
                                                const std::string& defaultCsvPath) {
    if (argc < 2 || argc > 4) {
        printUsage(argv[0], defaultCsvPath);
        return std::nullopt;
    }

    BenchmarkConfig config;
    config.csvPath = defaultCsvPath;

    try {
        // --- Tamaño del problema (obligatorio) ---------------------------
        // std::stoll permite detectar valores negativos antes de convertir
        // a std::size_t (una conversión directa los envolvería en silencio).
        const long long size = std::stoll(argv[1]);
        if (size <= 0) {
            std::cerr << "[ERROR] El tamano del problema debe ser > 0 "
                      << "(recibido: " << argv[1] << ").\n";
            printUsage(argv[0], defaultCsvPath);
            return std::nullopt;
        }
        config.problemSize = static_cast<std::size_t>(size);

        // --- Iteración (opcional) ----------------------------------------
        if (argc >= 3) {
            config.iteration = std::stoi(argv[2]);
            if (config.iteration < 1) {
                std::cerr << "[ERROR] La iteracion debe ser >= 1 "
                          << "(recibido: " << argv[2] << ").\n";
                printUsage(argv[0], defaultCsvPath);
                return std::nullopt;
            }
        }

        // --- Ruta del CSV (opcional) -------------------------------------
        if (argc >= 4) {
            config.csvPath = argv[3];
        }
    } catch (const std::exception&) {
        // std::stoll / std::stoi lanzan si el texto no es numérico o desborda.
        std::cerr << "[ERROR] Argumento numerico invalido.\n";
        printUsage(argv[0], defaultCsvPath);
        return std::nullopt;
    }

    return config;
}

}  // namespace bench
