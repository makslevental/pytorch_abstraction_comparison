//
// Created by Maksim Levental on 10/29/20.
//

#include <layers/activation.cuh>

/****************************************************************
 * Activation Layer                                             *
 ****************************************************************/

Activation::Activation(std::string name, cudnnActivationMode_t mode, double coef) {
    name_ = std::move(name);
    act_mode_ = mode;
    act_coef_ = coef;

    cudnnCreateActivationDescriptor(&act_desc_);
    cudnnSetActivationDescriptor(act_desc_, act_mode_, CUDNN_PROPAGATE_NAN, act_coef_);
}

Activation::~Activation() { cudnnDestroyActivationDescriptor(act_desc_); }

Tensor<double> *Activation::forward(Tensor<double> *input) {
    fwd_initialize(input);
    input_ = input;
    checkCudnnErrors(cudnnActivationForward(
        cuda_->cudnn(),
        act_desc_,
        &cuda_->one,
        input_desc_,
        input->get_device_ptr(),
        &cuda_->zero,
        output_desc_,
        output_->get_device_ptr()));

    return output_;
}

Tensor<double> *Activation::backward(Tensor<double> *grad_of_output) {
    bwd_initialize(grad_of_output);
    checkCudnnErrors(cudnnActivationBackward(
        cuda_->cudnn(),
        act_desc_,
        &cuda_->one,
        output_desc_,
        output_->get_device_ptr(),
        output_desc_,
        grad_of_output->get_device_ptr(),
        input_desc_,
        input_->get_device_ptr(),
        &cuda_->zero,
        input_desc_,
        grad_of_input_->get_device_ptr()));

    return grad_of_input_;
}
