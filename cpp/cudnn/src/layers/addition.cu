//
// Created by Maksim Levental on 11/5/20.
//

#include <layers/addition.cuh>

Tensor<double> *Addition::add(Tensor<double> *A, Tensor<double> *B) {
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
Tensor<double> *Addition::forward(Tensor<double> *input) { exit(EXIT_FAILURE); }
Tensor<double> *Addition::backward(Tensor<double> *grad_input) { exit(EXIT_FAILURE); }
void Addition::fwd_initialize(Tensor<double> *A) {
    Layer::fwd_initialize(A);
    if (op_descriptor == nullptr) {
        checkCudnnErrors(cudnnCreateOpTensorDescriptor(&op_descriptor));
        checkCudnnErrors(cudnnSetOpTensorDescriptor(
            op_descriptor, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_DOUBLE, CUDNN_PROPAGATE_NAN));
    }
}
Addition::~Addition() = default;
