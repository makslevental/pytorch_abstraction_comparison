#include <cross_entropy_loss.h>
#include <helper.h>

#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

/*
 * https://deepnotes.io/softmax-crossentropy
 * */

CrossEntropyLoss::CrossEntropyLoss(int batch_size, CudaContext *cuda_context) {
    checkCudaErrors(cudaMalloc((void **)&d_loss_, sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_workspace_, sizeof(double) * batch_size));
    if (op_descriptor_ == nullptr) {
        checkCudnnErrors(cudnnCreateOpTensorDescriptor(&op_descriptor_));
        checkCudnnErrors(cudnnSetOpTensorDescriptor(
            op_descriptor_, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_DOUBLE, CUDNN_PROPAGATE_NAN));
    }
    scale_ = 1.f / (double)batch_size;
    cuda_ = cuda_context;
}

CrossEntropyLoss::~CrossEntropyLoss() {
    if (d_loss_ != nullptr) {
        checkCudaErrors(cudaFree(d_loss_));
        d_loss_ = nullptr;
    }

    if (d_workspace_ != nullptr) {
        checkCudaErrors(cudaFree(d_workspace_));
        d_workspace_ = nullptr;
    }
}

__device__ double clip(double prediction, double epsilon = 1e-12) {
    return fmin(fmax(prediction, epsilon), 1.f - epsilon);
}

__global__ void cross_entropy_loss_kernel(
    double *reduced_loss,
    double *predict,
    double *target,
    double *workspace,
    int batch_size,
    int num_outputs) {
    unsigned int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

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
    for (unsigned int i = 0; i < batch_size; i += blockDim.x) {
        s_data[threadIdx.x] += workspace[threadIdx.x + i];
    }

    __syncthreads();

    // reduction
    // >>= is rightshift-assignment
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x + stride < batch_size)
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        reduced_loss[blockIdx.x] = s_data[0];
    }
}

double CrossEntropyLoss::loss(Tensor<double> *predict, Tensor<double> *target) {
    predict_ = predict;
    target_ = target;

    int num_sms;
    int num_blocks_per_sm;
    checkCudaErrors(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        cross_entropy_loss_kernel,
        BLOCK_DIM_1D,
        BLOCK_DIM_1D * sizeof(double)));

    int batch_size = target->get_batch_size();
    int num_outputs = target->get_channels();
    double h_loss_ = 0.f;

    cudaMemset(d_loss_, 0, sizeof(double));
    cudaMemset(d_workspace_, 0, sizeof(double) * batch_size);
    int num_blocks =
        std::min(num_blocks_per_sm * num_sms, (target->size() + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D);
    cross_entropy_loss_kernel<<<num_blocks, BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(double), 0>>>(
        d_loss_,
        predict->get_device_ptr(),
        target->get_device_ptr(),
        d_workspace_,
        batch_size,
        num_outputs);
    checkCudaErrors(cudaMemcpy(&h_loss_, d_loss_, sizeof(double), cudaMemcpyDeviceToHost));

    if (std::isnan(h_loss_)) {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if (std::isnan(h_loss_)) {
        std::cout << "[[ LOSS ]]" << std::endl;
        predict->print("predict", true, batch_size);
        target->print("target", true, batch_size);
        printf("h_loss_ %f\n", h_loss_);
        printf("h_loss_ / double(batch_size) %f\n", h_loss_ / double(batch_size));
        exit(EXIT_FAILURE);
    }
    // batch mean loss
    auto loss = h_loss_ / double(batch_size);
    return loss;
}

Tensor<double> *CrossEntropyLoss::backward() {
    if (grad_of_predict_ == nullptr)
        grad_of_predict_ = new Tensor<double>(predict_->shape());

    checkCudnnErrors(cudnnOpTensor(
        cuda_->cudnn(),
        op_descriptor_,
        &cuda_->one,
        predict_->tensor_descriptor(),
        predict_->get_device_ptr(),
        &cuda_->negative_one,
        target_->tensor_descriptor(),
        target_->get_device_ptr(),
        &cuda_->zero,
        grad_of_predict_->tensor_descriptor(),
        grad_of_predict_->get_device_ptr()));

    checkCublasErrors(cublasDscal(
        cuda_->cublas(), grad_of_predict_->len(), &scale_, grad_of_predict_->get_device_ptr(), 1));
    return grad_of_predict_;
}
