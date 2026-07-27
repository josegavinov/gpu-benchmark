# ============================================================================
# CMakeLists.txt — Benchmark Reduction (CUDA)
#
# Mismo esqueleto que cuda/vector_add/CMakeLists.txt (ver allí la
# justificación detallada de cada bloque): C++17, sm_50 (Quadro K2200),
# Release sin fast-math, includes desde la raíz del repositorio.
# ============================================================================
cmake_minimum_required(VERSION 3.18)

project(reduction LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Maxwell / Quadro K2200 = compute capability 5.0 (evita JIT en el arranque).
set(CMAKE_CUDA_ARCHITECTURES 50)

# Release ⇒ -O3. Sin --use_fast_math: rompería la comparabilidad IEEE 754.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(reduction
    main.cu
    kernel.cu
)

# Includes autoexplicativos desde la raíz del repositorio.
get_filename_component(GPU_BENCHMARK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
target_include_directories(reduction PRIVATE "${GPU_BENCHMARK_ROOT}")

target_compile_options(reduction PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
