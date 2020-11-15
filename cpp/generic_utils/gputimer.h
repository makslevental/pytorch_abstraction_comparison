//
// Created by Maksim Levental on 11/15/20.
//

#ifndef PYTORCH_ABSTRACTION_GPUTIMER_H
#define PYTORCH_ABSTRACTION_GPUTIMER_H

#include <cuda_runtime.h>

struct GpuTimer {

    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() const { cudaEventRecord(start_, nullptr); }

    void stop() const { cudaEventRecord(stop_, nullptr); }

    float elapsed() const {
        float elapsed;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed, start_, stop_);
        return elapsed;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

#endif // PYTORCH_ABSTRACTION_GPUTIMER_H
