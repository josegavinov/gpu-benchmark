# ============================================================================
# CMakeLists.txt — Benchmark Vector Addition (CUDA)
#
# CMake 3.18+ trae soporte de primera clase para CUDA como lenguaje
# (CMAKE_CUDA_ARCHITECTURES). Ubuntu 22.04 incluye CMake 3.22, así que el
# requisito se cumple sin instalar nada.
# ============================================================================
cmake_minimum_required(VERSION 3.18)

# Declarar CUDA como lenguaje hace que CMake localice nvcc (CUDA 11.8) y
# compile los .cu automáticamente con él.
project(vector_add LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------
# Estándares de lenguaje.
# C++17 es necesario para std::optional y std::filesystem usados en
# include/common/. CUDA 11.8 soporta C++17 tanto en host como en device.
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# Arquitectura GPU objetivo.
# La Quadro K2200 es Maxwell de primera generación = compute capability 5.0
# (sm_50). Compilar EXACTAMENTE para la arquitectura del experimento evita
# la compilación JIT en el primer lanzamiento (que distorsionaría tiempos)
# y genera código nativo óptimo.
# ----------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 50)

# ----------------------------------------------------------------------------
# Tipo de build.
# Release ⇒ -O3 para el código de host y optimización completa en nvcc.
# Un benchmark SIEMPRE se compila optimizado: medir un binario sin optimizar
# no dice nada del hardware. (Se puede forzar Debug con -DCMAKE_BUILD_TYPE.)
#
# NOTA: NO usamos --use_fast_math. Alteraría la semántica IEEE 754 de los
# flotantes y rompería la comparabilidad numérica con la CPU y con OpenCL.
# ----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ----------------------------------------------------------------------------
# Ejecutable.
# main.cu  : orquestador del experimento (fases 1-8)
# kernel.cu: kernel GPU + wrapper de lanzamiento
# ----------------------------------------------------------------------------
add_executable(vector_add
    main.cu
    kernel.cu
)

# ----------------------------------------------------------------------------
# Rutas de include.
# Se expone la RAÍZ del repositorio, de modo que los #include del código
# sean rutas autoexplicativas desde la raíz:
#     #include "include/common/csv_logger.hpp"   (compartido)
#     #include "cuda/common/cuda_utils.cuh"      (específico CUDA)
# Así se distingue de un vistazo qué es infraestructura común y qué no.
# ----------------------------------------------------------------------------
get_filename_component(GPU_BENCHMARK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
target_include_directories(vector_add PRIVATE "${GPU_BENCHMARK_ROOT}")

# ----------------------------------------------------------------------------
# Avisos del compilador al máximo para el código de host: en una
# infraestructura científica un warning es una futura fuente de dudas.
# (Se aplican solo al host: -Wall de GCC no es válido para el device.)
# ----------------------------------------------------------------------------
target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
