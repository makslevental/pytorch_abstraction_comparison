//
// Created by Maksim Levental on 11/5/20.
//

#include <layers/addition.cuh>

template <typename dtype> Addition<dtype>::~Addition() {}
template <typename dtype> Tensor<dtype> *Addition<dtype>::add(Tensor<dtype> *A, Tensor<dtype> *B) {
    fwd_initialize(A);
    // C = A + B
    checkCudnnErrors(cudnnOpTensor(
        this->cuda_->cudnn(),
        op_descriptor,
        &this->cuda_->one,
        A->tensor_descriptor(),
        A->get_device_ptr(),
        &this->cuda_->one,
        B->tensor_descriptor(),
        B->get_device_ptr(),
        &this->cuda_->zero,
        this->output_->tensor_descriptor(),
        this->output_->get_device_ptr()));

    return this->output_;
}
template <typename dtype> Tensor<dtype> *Addition<dtype>::forward(Tensor<dtype> *input) {
    exit(EXIT_FAILURE);
}
template <typename dtype> Tensor<dtype> *Addition<dtype>::backward(Tensor<dtype> *grad_input) {
    exit(EXIT_FAILURE);
}
template <typename dtype> void Addition<dtype>::fwd_initialize(Tensor<dtype> *A) {
    Layer<dtype>::fwd_initialize(A);
    if (op_descriptor == nullptr) {
        checkCudnnErrors(cudnnCreateOpTensorDescriptor(&op_descriptor));
        cudnnDataType_t t;
        if constexpr (std::is_same<dtype, float>{}) {
            t = CUDNN_DATA_FLOAT;
        } else if constexpr (std::is_same<dtype, double>{}) {
            t = CUDNN_DATA_DOUBLE;
        }
        checkCudnnErrors(
            cudnnSetOpTensorDescriptor(op_descriptor, CUDNN_OP_TENSOR_ADD, t, CUDNN_PROPAGATE_NAN));
    }
}
template <typename dtype> Addition<dtype>::Addition(std::string name) {
    this->name_ = std::move(name);
}
template class Addition<float>;
template class Addition<double>;
