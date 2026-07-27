# ============================================================================
# CMakeLists.txt — Benchmark Reduction (OpenCL)
#
# Mismo esqueleto que opencl/vector_add/CMakeLists.txt (ver allí la
# justificación): C++ puro en el host, kernel compilado en runtime por el
# driver, enlace contra el ICD loader (OpenCL::OpenCL).
# ============================================================================
cmake_minimum_required(VERSION 3.18)

project(reduction_opencl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release ⇒ -O3. Sin fast-math: semántica IEEE 754 intacta.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCL REQUIRED)

# Mismo nombre "reduction" que la versión CUDA: el directorio distingue.
add_executable(reduction
    main.cpp
    kernel.cpp
)

target_link_libraries(reduction PRIVATE OpenCL::OpenCL)

# Includes autoexplicativos desde la raíz del repositorio.
get_filename_component(GPU_BENCHMARK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
target_include_directories(reduction PRIVATE "${GPU_BENCHMARK_ROOT}")

target_compile_options(reduction PRIVATE -Wall -Wextra)
