//
// Created by Maksim Levental on 11/15/20.
//

#ifndef PYTORCH_ABSTRACTION_CUDA_PROFILING_H
#define PYTORCH_ABSTRACTION_CUDA_PROFILING_H

#include "cuda_helper.h"
#include "nvml.h"
#include <cuda_runtime.h>

struct GPUTimer {

    GPUTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() const { cudaEventRecord(start_, nullptr); }

    void stop() const { cudaEventRecord(stop_, nullptr); }

    [[nodiscard]] float elapsed() const {
        float elapsed;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed, start_, stop_);
        return elapsed;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

bool check_nvml(nvmlReturn_t r) {
    switch (r) {

    case NVML_SUCCESS:
        //        std::cout << "The operation was successful";
        return true;
    case NVML_ERROR_UNINITIALIZED:
        std::cout << "NVML was not first initialized with nvmlInit()";
        return false;
    case NVML_ERROR_INVALID_ARGUMENT:
        std::cout << "A supplied argument is invalid";
        return false;
    case NVML_ERROR_NOT_SUPPORTED:
        std::cout << "The requested operation is not available on target device";
        return false;
    case NVML_ERROR_NO_PERMISSION:
        std::cout << "The current user does not have permission for operation";
        return false;
    case NVML_ERROR_ALREADY_INITIALIZED:
        std::cout << "Deprecated: Multiple initializations are now allowed through ref counting";
        return false;
    case NVML_ERROR_NOT_FOUND:
        std::cout << "A query to find an object was unsuccessful";
        return false;
    case NVML_ERROR_INSUFFICIENT_SIZE:
        std::cout << "An input argument is not large enough";
        return false;
    case NVML_ERROR_INSUFFICIENT_POWER:
        std::cout << "A device's external power cables are not properly attached";
        return false;
    case NVML_ERROR_DRIVER_NOT_LOADED:
        std::cout << "NVIDIA driver is not loaded";
        return false;
    case NVML_ERROR_TIMEOUT:
        std::cout << "User provided timeout passed";
        return false;
    case NVML_ERROR_IRQ_ISSUE:
        std::cout << "NVIDIA Kernel detected an interrupt issue with a GPU";
        return false;
    case NVML_ERROR_LIBRARY_NOT_FOUND:
        std::cout << "NVML Shared Library couldn't be found or loaded";
        return false;
    case NVML_ERROR_FUNCTION_NOT_FOUND:
        std::cout << "Local version of NVML doesn't implement this function";
        return false;
    case NVML_ERROR_CORRUPTED_INFOROM:
        std::cout << "infoROM is corrupted";
        return false;
    case NVML_ERROR_GPU_IS_LOST:
        std::cout << "The GPU has fallen off the bus or has otherwise become inaccessible";
        return false;

    case NVML_ERROR_RESET_REQUIRED:
        std::cout << "The GPU requires a reset before it can be used again";
        return false;
    case NVML_ERROR_OPERATING_SYSTEM:
        std::cout << "The GPU control device has been blocked by the operating system/cgroups";
        return false;
    case NVML_ERROR_LIB_RM_VERSION_MISMATCH:
        std::cout << "RM detects a driver/library version mismatch";
        return false;
    case NVML_ERROR_IN_USE:
        std::cout << "An operation cannot be performed because the GPU is currently in use";
        return false;
    case NVML_ERROR_MEMORY:
        std::cout << "Insufficient memory";
        return false;
    case NVML_ERROR_NO_DATA:
        std::cout << "No data";
        return false;
    case NVML_ERROR_VGPU_ECC_NOT_SUPPORTED:
        std::cout << "The requested vgpu operation is not available on target device, becasue ECC "
                     "is enabled";
        return false;
    case NVML_ERROR_INSUFFICIENT_RESOURCES:
        std::cout << "Ran out of critical resources, other than memory ";
        return false;
    case NVML_ERROR_UNKNOWN:
        std::cout << "An internal driver error occurred";
        return false;
    default:
        std::cout << "unsupported flag";
        return false;
    }
}

bool nvml_inited = false;

int get_gpu_utilization() {
    if (!nvml_inited) {
        if (!check_nvml(nvmlInit_v2())) {
            exit(EXIT_FAILURE);
        }
    }
    nvmlDevice_t device;
    auto device_id = atoi(std::getenv("CUDA_VISIBLE_DEVICES"));
    nvmlDeviceGetHandleByIndex_v2(device_id, &device);
    nvmlUtilization_t util;
    if (!check_nvml(nvmlDeviceGetUtilizationRates(device, &util))) {
        exit(EXIT_FAILURE);
    }
    return util.gpu;
}

static double get_used_cuda_mem() {
    // show memory usage of GPU
    size_t free_byte;
    size_t total_byte;
    checkCudaErrors(cudaMemGetInfo(&free_byte, &total_byte));
    auto free_db = (double)free_byte;
    auto total_db = (double)total_byte;
    double used_db = total_db - free_db;
    return used_db / 1024.0 / 1024.0;
}

#endif // PYTORCH_ABSTRACTION_CUDA_PROFILING_H
