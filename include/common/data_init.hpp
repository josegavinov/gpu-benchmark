/**
 * @file data_init.hpp
 * @brief Inicialización determinista de datos de entrada.
 *
 * PRINCIPIO CIENTÍFICO — REPRODUCIBILIDAD:
 * Los datos de entrada se generan con un PRNG de semilla FIJA (Mersenne
 * Twister). Así, cualquier ejecución —hoy en CUDA, mañana en OpenCL, o por
 * un revisor del artículo dentro de un año— opera sobre EXACTAMENTE los
 * mismos valores. Esto elimina la entrada como variable de confusión en la
 * comparación entre frameworks.
 *
 * NO se usa rand()/srand(): su secuencia depende de la implementación de la
 * libc, lo que rompería la reproducibilidad entre máquinas. std::mt19937
 * está completamente especificado por el estándar C++ y produce la misma
 * secuencia en cualquier plataforma.
 *
 * Módulo C++ puro: compartido entre CUDA y OpenCL.
 */

#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace bench {

/// Semillas fijas del protocolo experimental. Cada operando usa una semilla
/// distinta para evitar correlación artificial entre A y B (A[i] == B[i]).
inline constexpr unsigned kSeedVectorA = 42u;
inline constexpr unsigned kSeedVectorB = 1337u;

/**
 * @brief Rellena un vector con valores pseudoaleatorios reproducibles.
 *
 * @param data     Vector destino (se redimensiona a `size`).
 * @param size     Número de elementos a generar.
 * @param seed     Semilla del generador (fija por protocolo).
 * @param minValue Límite inferior del rango uniforme.
 * @param maxValue Límite superior del rango uniforme.
 *
 * El rango por defecto [0, 1] mantiene los valores en una zona numérica
 * "cómoda" para float: sumar valores de magnitud similar minimiza el error
 * de redondeo y simplifica el análisis de la verificación.
 */
inline void fillRandom(std::vector<float>& data,
                       std::size_t size,
                       unsigned seed,
                       float minValue = 0.0f,
                       float maxValue = 1.0f) {
    data.resize(size);

    std::mt19937 generator(seed);  // PRNG determinista y portable.
    std::uniform_real_distribution<float> distribution(minValue, maxValue);

    for (std::size_t i = 0; i < size; ++i) {
        data[i] = distribution(generator);
    }
}

}  // namespace bench
