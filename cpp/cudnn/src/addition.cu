//
// Created by Maksim Levental on 11/5/20.
//

#include <addition.cuh>

Tensor<float> *Addition::add(Tensor<float> *A, Tensor<float> *B) {
    fwd_initialize(A);
    // C = A + B
    checkCudnnErrors(cudnnOpTensor(
        cuda_->cudnn(),
        op_descriptor,
        &cuda_->one,
        A->tensor_descriptor(),
        A->get_device_ptr(),
        &cuda_->one,
        B->tensor_descriptor(),
        B->get_device_ptr(),
        &cuda_->zero,
        output_->tensor_descriptor(),
        output_->get_device_ptr()));

    return output_;
}
Tensor<float> *Addition::forward(Tensor<float> *input) { exit(EXIT_FAILURE); }
Tensor<float> *Addition::backward(Tensor<float> *grad_input) { exit(EXIT_FAILURE); }
void Addition::fwd_initialize(Tensor<float> *A) {
    Layer::fwd_initialize(A);
    if (op_descriptor == nullptr) {
        checkCudnnErrors(cudnnCreateOpTensorDescriptor(&op_descriptor));
        checkCudnnErrors(cudnnSetOpTensorDescriptor(
            op_descriptor, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
    }
}
Addition::~Addition() = default;
