/**
 * @file opencl_utils.hpp
 * @brief Utilidades OpenCL compartidas — espejo de cuda/common/cuda_utils.cuh.
 *
 * CORRESPONDENCIA CON LA VERSIÓN CUDA (simetría experimental):
 *
 *   cuda_utils.cuh                    opencl_utils.hpp
 *   ------------------------------    -----------------------------------
 *   CUDA_CHECK(expr)                  CL_CHECK(expr)
 *   EventTimer (CUDA Events)          eventElapsedMs(cl_event) (profiling)
 *   printDeviceInfo()                 printDeviceInfo(device)
 *   (contexto implícito de CUDA)      ClEnvironment (explícito en OpenCL)
 *
 * DIFERENCIA CONCEPTUAL CLAVE — y parte del objeto de estudio:
 * CUDA gestiona plataforma/contexto/cola de forma implícita (el runtime crea
 * todo en la primera llamada). OpenCL lo hace EXPLÍCITO: hay que descubrir
 * la plataforma, elegir el dispositivo, crear el contexto y la cola a mano.
 * Ese trabajo extra es parte del "costo de framework" que la investigación
 * compara; aquí queda encapsulado en ClEnvironment para que main.cpp
 * conserve la misma legibilidad que main.cu.
 */

#pragma once

// Versión objetivo del API. El driver NVIDIA 535 expone OpenCL 3.0; fijarlo
// explícitamente evita los warnings de "versión no definida" de cl.h y
// habilita clCreateCommandQueueWithProperties (necesaria para profiling).
#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace bench {
namespace ocl {

/**
 * @brief Traduce un código de error OpenCL a texto legible.
 *
 * OpenCL reporta errores como enteros negativos (cl_int). Se cubren los
 * códigos que este proyecto puede encontrar en la práctica; cualquier otro
 * se muestra numérico (suficiente para buscarlo en cl.h).
 */
inline const char* clErrorString(cl_int code) {
    switch (code) {
        case CL_SUCCESS:                        return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:               return "CL_DEVICE_NOT_FOUND";
        case CL_OUT_OF_RESOURCES:               return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:             return "CL_OUT_OF_HOST_MEMORY";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:  return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_BUILD_PROGRAM_FAILURE:          return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE:                  return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE:                 return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE:          return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT:             return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_KERNEL:                 return "CL_INVALID_KERNEL";
        case CL_INVALID_KERNEL_ARGS:            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_GROUP_SIZE:        return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_GLOBAL_WORK_SIZE:       return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_PROFILING_INFO_NOT_AVAILABLE:   return "CL_PROFILING_INFO_NOT_AVAILABLE";
        default:                                return "(ver cl.h)";
    }
}

/**
 * @brief Verificación obligatoria de toda llamada OpenCL — espejo de CUDA_CHECK.
 *
 * MISMA POLÍTICA QUE EN CUDA: en un benchmark, un error NO puede ignorarse
 * (invalidaría la medición en silencio). Ante cualquier fallo: diagnóstico
 * con archivo/línea y terminación inmediata.
 */
#define CL_CHECK(expr)                                                        \
    do {                                                                      \
        const cl_int cl_check_status_ = (expr);                               \
        if (cl_check_status_ != CL_SUCCESS) {                                 \
            std::fprintf(stderr,                                              \
                         "[OpenCL ERROR] %s (%d)\n  en %s:%d\n  expr: %s\n",  \
                         bench::ocl::clErrorString(cl_check_status_),         \
                         cl_check_status_, __FILE__, __LINE__, #expr);        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

/**
 * @brief Entorno OpenCL: plataforma → dispositivo → contexto → cola.
 *
 * Encapsula el "boilerplate" de descubrimiento que CUDA no necesita.
 * Selecciona el PRIMER dispositivo de tipo GPU disponible (en el servidor
 * experimental solo existe la Quadro K2200 vía el ICD de NVIDIA, así que la
 * selección es determinista).
 *
 * La cola se crea con CL_QUEUE_PROFILING_ENABLE: sin esa propiedad, los
 * eventos no registran marcas de tiempo y clGetEventProfilingInfo fallaría.
 */
class ClEnvironment {
public:
    ClEnvironment() {
        // --- Plataforma y dispositivo -----------------------------------
        cl_uint platformCount = 0;
        CL_CHECK(clGetPlatformIDs(0, nullptr, &platformCount));
        if (platformCount == 0) {
            std::fprintf(stderr, "[OpenCL ERROR] No hay plataformas OpenCL.\n");
            std::exit(EXIT_FAILURE);
        }
        std::vector<cl_platform_id> platforms(platformCount);
        CL_CHECK(clGetPlatformIDs(platformCount, platforms.data(), nullptr));

        // Recorrer plataformas hasta encontrar una GPU.
        for (cl_platform_id p : platforms) {
            cl_uint deviceCount = 0;
            const cl_int status =
                clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device_, &deviceCount);
            if (status == CL_SUCCESS && deviceCount > 0) {
                platform_ = p;
                break;
            }
        }
        if (device_ == nullptr) {
            std::fprintf(stderr, "[OpenCL ERROR] No se encontro ninguna GPU.\n");
            std::exit(EXIT_FAILURE);
        }

        // --- Contexto y cola con profiling -------------------------------
        cl_int status = CL_SUCCESS;
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &status);
        CL_CHECK(status);

        const cl_queue_properties props[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &status);
        CL_CHECK(status);
    }

    /// Liberación en orden inverso a la creación (regla general en OpenCL).
    ~ClEnvironment() {
        if (queue_)   clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
        // Los cl_device_id de GetDeviceIDs no se liberan (no son "retained").
    }

    // El entorno posee recursos únicos del proceso: prohibir copias.
    ClEnvironment(const ClEnvironment&)            = delete;
    ClEnvironment& operator=(const ClEnvironment&) = delete;

    cl_device_id     device()  const { return device_; }
    cl_context       context() const { return context_; }
    cl_command_queue queue()   const { return queue_; }

private:
    cl_platform_id   platform_ = nullptr;
    cl_device_id     device_   = nullptr;
    cl_context       context_  = nullptr;
    cl_command_queue queue_    = nullptr;
};

/**
 * @brief Tiempo de GPU de un comando, en ms — espejo de EventTimer::elapsedMs().
 *
 * Los eventos de profiling de OpenCL registran marcas de tiempo del RELOJ DE
 * LA GPU (nanosegundos) al inicio (COMMAND_START) y fin (COMMAND_END) de la
 * ejecución real del comando — el equivalente directo de cudaEventElapsedTime.
 *
 * Se espera al evento (clWaitForEvents) antes de leer: igual que en CUDA,
 * la medición es del trabajo TERMINADO, no del encolado.
 */
inline double eventElapsedMs(cl_event event) {
    CL_CHECK(clWaitForEvents(1, &event));
    cl_ulong startNs = 0;
    cl_ulong endNs   = 0;
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                     sizeof(startNs), &startNs, nullptr));
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                     sizeof(endNs), &endNs, nullptr));
    return static_cast<double>(endNs - startNs) / 1.0e6;  // ns → ms
}

/**
 * @brief Traza del hardware por stderr — espejo de printDeviceInfo() de CUDA.
 *
 * Deja constancia en cada corrida de QUÉ dispositivo y versión de driver
 * midieron: trazabilidad básica del protocolo experimental.
 */
inline void printDeviceInfo(cl_device_id device) {
    char name[256]    = {0};
    char version[256] = {0};
    char driver[256]  = {0};
    cl_ulong globalMem = 0;
    cl_uint  computeUnits = 0;

    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, nullptr));
    CL_CHECK(clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(driver), driver, nullptr));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                             sizeof(globalMem), &globalMem, nullptr));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                             sizeof(computeUnits), &computeUnits, nullptr));

    std::fprintf(stderr, "[INFO] GPU: %s | %s | %llu MB | %u CUs | Driver %s\n",
                 name, version,
                 static_cast<unsigned long long>(globalMem / (1024 * 1024)),
                 computeUnits, driver);
}

}  // namespace ocl
}  // namespace bench
