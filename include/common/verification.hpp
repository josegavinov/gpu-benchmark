/**
 * @file verification.hpp
 * @brief Verificación funcional de los resultados de la GPU.
 *
 * PRINCIPIO CIENTÍFICO — VALIDEZ:
 * Un tiempo medido solo es válido si el cómputo fue correcto. Por eso cada
 * ejecución compara el 100% de los elementos calculados en GPU contra una
 * referencia calculada en CPU, y el veredicto (PASS/FAIL) acompaña a la
 * medición en el CSV. Una fila FAIL se descarta del análisis estadístico.
 *
 * COMPARACIÓN CON TOLERANCIA:
 * Aunque para C[i] = A[i] + B[i] la GPU y la CPU deberían producir el mismo
 * resultado bit a bit (misma operación IEEE 754 en float), se usa una
 * tolerancia relativa configurable. Motivo: esta misma función se reutilizará
 * en benchmarks donde el orden de las operaciones difiere entre CPU y GPU
 * (Reduction, MatrixMul), y allí la igualdad exacta es matemáticamente
 * inalcanzable en punto flotante. Un único mecanismo de verificación para
 * todos los benchmarks evita duplicación (DRY).
 *
 * Módulo C++ puro: compartido entre CUDA y OpenCL.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

namespace bench {

/**
 * @brief Veredicto de la verificación, con detalle del primer error hallado.
 */
struct VerificationResult {
    bool passed = true;           ///< true si TODOS los elementos coinciden.
    std::size_t errorIndex = 0;   ///< Posición del primer elemento erróneo.
    float expectedValue = 0.0f;   ///< Valor de referencia (CPU) en esa posición.
    float obtainedValue = 0.0f;   ///< Valor calculado (GPU) en esa posición.

    /// Estado en el formato exigido por la columna "Status" del CSV.
    std::string status() const { return passed ? "PASS" : "FAIL"; }
};

/**
 * @brief Compara elemento a elemento el resultado de la GPU con la referencia.
 *
 * @param reference         Resultado de referencia calculado en CPU.
 * @param result            Resultado calculado en GPU.
 * @param size              Número de elementos a comparar (todos).
 * @param relativeTolerance Tolerancia relativa admitida (por defecto 1e-5,
 *                          holgada para float pero muy inferior a cualquier
 *                          error real de cómputo).
 *
 * Criterio de aceptación estándar en HPC:
 *     |ref - res| <= tol * max(|ref|, 1)
 * El término max(...,1) actúa como tolerancia absoluta cuando la referencia
 * es cercana a cero, donde la tolerancia puramente relativa degenera.
 *
 * Se recorre TODO el vector (sin salida temprana en el primer acierto) pero
 * se informa únicamente del PRIMER error, como exige el protocolo.
 */
inline VerificationResult verifyElementwise(const float* reference,
                                            const float* result,
                                            std::size_t size,
                                            float relativeTolerance = 1e-5f) {
    VerificationResult verdict;

    for (std::size_t i = 0; i < size; ++i) {
        const float difference = std::fabs(reference[i] - result[i]);
        const float tolerance =
            relativeTolerance * std::max(std::fabs(reference[i]), 1.0f);

        if (difference > tolerance) {
            verdict.passed        = false;
            verdict.errorIndex    = i;
            verdict.expectedValue = reference[i];
            verdict.obtainedValue = result[i];
            return verdict;  // Primer error: suficiente para diagnosticar.
        }
    }

    return verdict;  // passed == true
}

/**
 * @brief Imprime el veredicto en consola en el formato exigido (PASS/FAIL).
 *
 * En caso de fallo indica posición, valor esperado y valor obtenido, tal
 * como requiere el protocolo experimental.
 */
inline void printVerification(const VerificationResult& verdict) {
    if (verdict.passed) {
        std::cout << "PASS\n";
    } else {
        std::cout << "FAIL\n"
                  << "  Posicion : " << verdict.errorIndex << '\n'
                  << "  Esperado : " << verdict.expectedValue << '\n'
                  << "  Obtenido : " << verdict.obtainedValue << '\n';
    }
}

}  // namespace bench
