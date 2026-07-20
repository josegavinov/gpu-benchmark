# ============================================================================
# CMakeLists.txt — Benchmark Vector Addition (OpenCL)
#
# Espejo del CMakeLists.txt de la versión CUDA, con una diferencia
# estructural que es en sí misma un dato del estudio: aquí NO hay compilador
# de dispositivo. OpenCL es C++ puro en el host; el kernel se compila en
# RUNTIME (driver). Por eso basta el compilador C++ del sistema y enlazar
# contra la librería OpenCL (el ICD loader).
# ============================================================================
cmake_minimum_required(VERSION 3.18)

project(vector_add_opencl LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Estándar de lenguaje: el mismo C++17 que la versión CUDA (por
# std::optional y std::filesystem de include/common/).
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# Tipo de build: Release (-O3), mismo criterio que en CUDA — un benchmark
# solo tiene sentido optimizado. Sin fast-math: semántica IEEE 754 intacta
# para la comparabilidad numérica (el kernel tampoco usa flags relajados).
# ----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ----------------------------------------------------------------------------
# Localizar OpenCL: headers (opencl-headers) + ICD loader (ocl-icd-opencl-dev).
# El módulo FindOpenCL de CMake define el target importado OpenCL::OpenCL
# con includes y librería ya configurados.
# ----------------------------------------------------------------------------
find_package(OpenCL REQUIRED)

# ----------------------------------------------------------------------------
# Ejecutable. Mismo nombre "vector_add" que la versión CUDA: el directorio
# (opencl/ vs cuda/) es quien distingue, y los scripts reciben la ruta.
# ----------------------------------------------------------------------------
add_executable(vector_add
    main.cpp
    kernel.cpp
)

target_link_libraries(vector_add PRIVATE OpenCL::OpenCL)

# ----------------------------------------------------------------------------
# Rutas de include: la raíz del repositorio, igual que en CUDA, para que
# los #include sean autoexplicativos:
#     #include "include/common/csv_logger.hpp"     (compartido)
#     #include "opencl/common/opencl_utils.hpp"    (específico OpenCL)
# ----------------------------------------------------------------------------
get_filename_component(GPU_BENCHMARK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
target_include_directories(vector_add PRIVATE "${GPU_BENCHMARK_ROOT}")

# ----------------------------------------------------------------------------
# Avisos al máximo, mismo criterio que la versión CUDA.
# ----------------------------------------------------------------------------
target_compile_options(vector_add PRIVATE -Wall -Wextra)
