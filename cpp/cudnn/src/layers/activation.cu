//
// Created by Maksim Levental on 10/29/20.
//

#include <layers/activation.cuh>

/****************************************************************
 * Activation Layer                                             *
 ****************************************************************/

template <typename dtype>
Activation<dtype>::Activation(std::string name, cudnnActivationMode_t mode, double coef) {
    this->name_ = std::move(name);
    act_mode_ = mode;
    act_coef_ = coef;

    cudnnCreateActivationDescriptor(&act_desc_);
    cudnnSetActivationDescriptor(act_desc_, act_mode_, CUDNN_PROPAGATE_NAN, act_coef_);
}

template <typename dtype> Activation<dtype>::~Activation() {
    cudnnDestroyActivationDescriptor(act_desc_);
}
template <typename dtype> Tensor<dtype> *Activation<dtype>::forward(Tensor<dtype> *input) {
    this->fwd_initialize(input);
    this->input_ = input;
    checkCudnnErrors(cudnnActivationForward(
        this->cuda_->cudnn(),
        act_desc_,
        &this->cuda_->one,
        this->input_desc_,
        input->get_device_ptr(),
        &this->cuda_->zero,
        this->output_desc_,
        this->output_->get_device_ptr()));

    return this->output_;
}
template <typename dtype>
Tensor<dtype> *Activation<dtype>::backward(Tensor<dtype> *grad_of_output) {
    this->bwd_initialize(grad_of_output);
    checkCudnnErrors(cudnnActivationBackward(
        this->cuda_->cudnn(),
        act_desc_,
        &this->cuda_->one,
        this->output_desc_,
        this->output_->get_device_ptr(),
        this->output_desc_,
        grad_of_output->get_device_ptr(),
        this->input_desc_,
        this->input_->get_device_ptr(),
        &this->cuda_->zero,
        this->input_desc_,
        this->grad_of_input_->get_device_ptr()));

    return this->grad_of_input_;
}

template class Activation<float>;

template class Activation<double>;
