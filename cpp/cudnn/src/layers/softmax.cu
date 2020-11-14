//
// Created by Maksim Levental on 10/29/20.
//

#include <cassert>
#include <layers/softmax.cuh>

Softmax::Softmax(std::string name) { name_ = std::move(name); }

Tensor<double> *Softmax::forward(Tensor<double> *input) {
    fwd_initialize(input);
    input_ = input;
    if (DEBUG_SOFTMAX) {
        std::cout << name_ << "[FORWARD]" << std::endl;
        input->print(name_ + "::input", true, input->get_batch_size());
    }
    checkCudnnErrors(cudnnSoftmaxForward(
        cuda_->cudnn(),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &cuda_->one,
        input_desc_,
        input->get_device_ptr(),
        &cuda_->zero,
        output_desc_,
        output_->get_device_ptr()));

    if (DEBUG_SOFTMAX)
        output_->print(name_ + "::output", true, output_->get_batch_size());

    return output_;
}

Tensor<double> *Softmax::backward(Tensor<double> *grad_of_output) {
    bwd_initialize(grad_of_output);

    checkCudnnErrors(cudnnSoftmaxBackward(
        cuda_->cudnn(),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &cuda_->one,
        output_desc_,
        output_->get_device_ptr(),
        grad_of_output->tensor_descriptor(),
        grad_of_output->get_device_ptr(),
        &cuda_->zero,
        grad_of_input_->tensor_descriptor(),
        grad_of_input_->get_device_ptr()));

    if (DEBUG_SOFTMAX > 1) {
        std::cout << name_ << "[BACKWARD]" << std::endl;
        output_->print(name_ + "::predict", true);
        grad_of_output->print(name_ + "::y", true, grad_of_output->get_batch_size());
        grad_of_input_->print(name_ + "::dx", true, grad_of_output->get_batch_size());
    }

    return grad_of_input_;
}
