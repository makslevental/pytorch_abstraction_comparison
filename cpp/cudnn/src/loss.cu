#include <helper.h>
#include <loss.h>

#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

/*
 * https://deepnotes.io/softmax-crossentropy
 * */

CrossEntropyLoss::CrossEntropyLoss() {
    checkCudaErrors(cudaMalloc((void **)&d_loss_, sizeof(double)));
}

CrossEntropyLoss::~CrossEntropyLoss() {
    if (d_loss_ != nullptr) {
        checkCudaErrors(cudaFree(d_loss_));
        d_loss_ = nullptr;
    }

    if (d_workspace_ != nullptr)
        checkCudaErrors(cudaFree(d_workspace_));
}

__device__ double clip(double prediction, double epsilon = 1e-12) {
    return fmin(fmax(prediction, epsilon), 1.f - epsilon);
}

__global__ void softmax_loss_kernel(
    double *reduced_loss,
    double *predict,
    double *target,
    double *workspace,
    int batch_size,
    int num_outputs) {
    int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ double s_data[];
    double loss = 0.f;

    // each thread calculate entropy for each data and accumulate to shared memory
    for (int c = 0; c < num_outputs; c++)
        loss += target[batch_idx * num_outputs + c] * logb(predict[batch_idx * num_outputs + c]);
    workspace[batch_idx] = -loss;

    // then, we do reduction the result to calculate loss using 1 thread block
    if (blockIdx.x > 0)
        return;

    // cumulate workspace data
    s_data[threadIdx.x] = 0.f;
    for (int i = 0; i < batch_size; i += blockDim.x) {
        s_data[threadIdx.x] += workspace[threadIdx.x + i];
    }

    __syncthreads();

    // reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x + stride < batch_size)
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        reduced_loss[blockIdx.x] = s_data[0];
    }
}

void CrossEntropyLoss::init_workspace(int batch_size) {
    if (d_workspace_ == nullptr)
        checkCudaErrors(cudaMalloc((void **)&d_workspace_, sizeof(double) * batch_size));
}

double CrossEntropyLoss::loss(Tensor<double> *predict, Tensor<double> *target) {
    int num_sms;
    int num_blocks_per_sm;
    checkCudaErrors(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm, softmax_loss_kernel, BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(double)));

    int batch_size = target->get_batch_size();
    int num_outputs = target->get_channels();

    init_workspace(batch_size);

    if (DEBUG_LOSS) {
        std::cout << "[[ LOSS ]]" << std::endl;
        predict->print("predict", true);
        target->print("target", true);
    }

    int num_blocks =
        std::min(num_blocks_per_sm * num_sms, (target->size() + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D);
    softmax_loss_kernel<<<num_blocks, BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(double), 0>>>(
        d_loss_,
        predict->get_device_ptr(),
        target->get_device_ptr(),
        d_workspace_,
        batch_size,
        num_outputs);
    checkCudaErrors(cudaMemcpy(&h_loss_, d_loss_, sizeof(double), cudaMemcpyDeviceToHost));

    // batch mean loss
    auto loss = h_loss_ / double(batch_size);
    return loss;
}