//
// Created by Maksim Levental on 10/29/20.
//

#include <activation.cuh>

/****************************************************************
 * Activation Layer                                             *
 ****************************************************************/

Activation::Activation(std::string name, cudnnActivationMode_t mode, float coef) {
    name_ = std::move(name);
    act_mode_ = mode;
    act_coef_ = coef;

    cudnnCreateActivationDescriptor(&act_desc_);
    cudnnSetActivationDescriptor(act_desc_, act_mode_, CUDNN_PROPAGATE_NAN, act_coef_);
}

Activation::~Activation() { cudnnDestroyActivationDescriptor(act_desc_); }

void Activation::fwd_initialize(Tensor<float> *input) {
    if (input_desc_ == nullptr || batch_size_ != input->get_batch_size()) {
        //        input_ = input;
        input_size_ = input->size();
        input_desc_ = input->tensor_descriptor();
        batch_size_ = input->get_batch_size();

        if (output_ == nullptr)
            output_ = new Tensor<float>(input->shape());
        else
            output_->reset(input->shape());

        output_desc_ = output_->tensor_descriptor();
    }
}

Tensor<float> *Activation::forward(Tensor<float> *input) {
    fwd_initialize(input);
    input_ = input;
    cudnnActivationForward(
        cuda_->cudnn(),
        act_desc_,
        &cuda_->one,
        input_desc_,
        input->get_device_ptr(),
        &cuda_->zero,
        output_desc_,
        output_->get_device_ptr());

    return output_;
}

Tensor<float> *Activation::backward(Tensor<float> *grad_output) {
    bwd_initialize(grad_output);
    cudnnActivationBackward(
        cuda_->cudnn(),
        act_desc_,
        &cuda_->one,
        output_desc_,
        output_->get_device_ptr(),
        output_desc_,
        grad_output->get_device_ptr(),
        input_desc_,
        input_->get_device_ptr(),
        &cuda_->zero,
        input_desc_,
        grad_input_->get_device_ptr());

    return grad_input_;
}
