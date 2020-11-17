//
// Created by Maksim Levental on 11/15/20.
//

#ifndef PYTORCH_ABSTRACTION_GPUTIMER_H
#define PYTORCH_ABSTRACTION_GPUTIMER_H

#include <cuda_runtime.h>
#include<nvml.h>

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

//int get_utilization(){
//    nvmlDevice_t d;
//    nvmlDeviceGetHandleByIndex_v2
//    nvmlUtilization_t util;
//    nvmlDeviceGetUtilizationRates()
//}

#endif // PYTORCH_ABSTRACTION_GPUTIMER_H
