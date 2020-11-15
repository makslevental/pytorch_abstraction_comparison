#ifndef _HELPER_H_
#define _HELPER_H_

#include <cublas_v2.h>
#include <cudnn.h>

#include "stacktrace.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#include <fstream>
#include <iostream>
#include <prettyprint.h>
#include <sstream>
#include <utility>

// #include <helper_cuda.h>
#include <curand.h>

#define BLOCK_DIM_1D 512
#define BLOCK_DIM 16

/* DEBUG FLAGS */
#define DEBUG_FORWARD 0
#define DEBUG_BACKWARD 0

#define DEBUG_CONV 0
#define DEBUG_DENSE 0
#define DEBUG_SOFTMAX 0
#define DEBUG_UPDATE 0

#define DEBUG_LOSS 0
#define DEBUG_ACCURACY 0

#define DEBUG_FIND_ALGO 0

/* CUDA API error return checker */
#ifndef checkCudaErrors
#define checkCudaErrors(err)                                                                       \
    do {                                                                                           \
        if (err != cudaSuccess) {                                                                  \
            fprintf(                                                                               \
                stderr,                                                                            \
                "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n",            \
                err,                                                                               \
                cudaGetErrorString(err),                                                           \
                __FILE__,                                                                          \
                __LINE__);                                                                         \
            fprintf(stderr, "%d\n", cudaSuccess);                                                  \
            print_stacktrace();                                                                    \
            exit(-1);                                                                              \
        }                                                                                          \
    } while (0)
#endif

static const char *_cublasGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

#define checkCublasErrors(err)                                                                     \
    do {                                                                                           \
        if (err != CUBLAS_STATUS_SUCCESS) {                                                        \
            fprintf(                                                                               \
                stderr,                                                                            \
                "checkCublasErrors() API error = %04d \"%s\" from file <%s>, line %i.\n",          \
                err,                                                                               \
                _cublasGetErrorEnum(err),                                                          \
                __FILE__,                                                                          \
                __LINE__);                                                                         \
            print_stacktrace();                                                                    \
            exit(-1);                                                                              \
        }                                                                                          \
    } while (0)

#define checkCudnnErrors(err)                                                                      \
    do {                                                                                           \
        if (err != CUDNN_STATUS_SUCCESS) {                                                         \
            fprintf(                                                                               \
                stderr,                                                                            \
                "checkCudnnErrors() API error = %04d \"%s\" from file <%s>, line %i.\n",           \
                err,                                                                               \
                cudnnGetErrorString(err),                                                          \
                __FILE__,                                                                          \
                __LINE__);                                                                         \
            print_stacktrace();                                                                    \
            exit(-1);                                                                              \
        }                                                                                          \
    } while (0)

// cuRAND API errors
static const char *_curandGetErrorEnum(curandStatus_t error) {
    switch (error) {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define checkCurandErrors(err)                                                                     \
    {                                                                                              \
        if (err != CURAND_STATUS_SUCCESS) {                                                        \
            fprintf(                                                                               \
                stderr,                                                                            \
                "checkCurandErrors() API error = %04d \"%s\" from file <%s>, line %i.\n",          \
                err,                                                                               \
                _curandGetErrorEnum(err),                                                          \
                __FILE__,                                                                          \
                __LINE__);                                                                         \
            exit(-1);                                                                              \
        }                                                                                          \
    }

#define PRINT_LINE std::cout << __FILE__ << ":" << __LINE__ << std::endl;
#define PRINT(x) std::cout << x << std::endl;

// container for get_device_ptr resources
template <typename dtype> class CudaContext {
public:
    CudaContext() {
        cublasCreate(&_cublas_handle);
        checkCudaErrors(cudaGetLastError());
        checkCudnnErrors(cudnnCreate(&_cudnn_handle));
    }
    ~CudaContext() {
        cublasDestroy(_cublas_handle);
        checkCudnnErrors(cudnnDestroy(_cudnn_handle));
    }

    cublasHandle_t cublas() {
        // std::cout << "Get cublas request" << std::endl; getchar();
        return _cublas_handle;
    };
    cudnnHandle_t cudnn() { return _cudnn_handle; };

    const dtype one = 1.f;
    const dtype zero = 0.f;
    const dtype minus_one = -1.f;

private:
    cublasHandle_t _cublas_handle;
    cudnnHandle_t _cudnn_handle;
};

static void print_tensor_descriptor(const std::string &name, cudnnTensorDescriptor_t t) {
    cudnnDataType_t dataType;
    const int nbDimsRequested = 10;
    int nbDims;
    int dimA[nbDimsRequested] = {};
    int strideA[nbDimsRequested] = {};
    checkCudnnErrors(
        cudnnGetTensorNdDescriptor(t, nbDimsRequested, &dataType, &nbDims, dimA, strideA));
    std::array<int, nbDimsRequested> dimA_arr{};
    std::copy(std::begin(dimA), std::end(dimA), std::begin(dimA_arr));
    std::array<int, nbDimsRequested> strideA_arr{};
    std::copy(std::begin(strideA), std::end(strideA), std::begin(strideA_arr));
    std::cout << name << " datatype: " << dataType << " nbDims: " << nbDims << " dimA: " << dimA_arr
              << " strideA: " << strideA_arr << std::endl;
}

static int to_int(const uint8_t *ptr) {
    return (
        (ptr[0] & 0xFF) << 24 | (ptr[1] & 0xFF) << 16 | (ptr[2] & 0xFF) << 8 |
        (ptr[3] & 0xFF) << 0);
}

static double get_used_cuda_mem() {
    // show memory usage of GPU
    size_t free_byte;
    size_t total_byte;
    checkCudaErrors(cudaMemGetInfo(&free_byte, &total_byte));
    auto free_db = (double)free_byte;
    auto total_db = (double)total_byte;
    double used_db = total_db - free_db;
//    printf(
//        "GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
//        used_db / 1024.0 / 1024.0,
//        free_db / 1024.0 / 1024.0,
//        total_db / 1024.0 / 1024.0);
    return used_db;
}

#endif // _HELPER_H_