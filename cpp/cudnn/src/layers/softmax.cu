//
// Created by Maksim Levental on 10/29/20.
//

#include <cassert>
#include <layers/softmax.cuh>

Softmax::Softmax(std::string name) { name_ = std::move(name); }

void Softmax::fwd_initialize(Tensor<float> *input) {
    if (input_desc_ == nullptr || batch_size_ != input->get_batch_size()) {
        input_desc_ = input->tensor_descriptor();
        batch_size_ = input->get_batch_size();

        if (output_ == nullptr)
            output_ = new Tensor<float>(input->shape());
        else
            output_->reset(input->shape());

        output_desc_ = output_->tensor_descriptor();
    }
}

Tensor<float> *Softmax::forward(Tensor<float> *input) {
    fwd_initialize(input);
    if (DEBUG_SOFTMAX) {
        std::cout << name_ << "[FORWARD]" << std::endl;
        input_->print(name_ + "::input", true, input->get_batch_size());
    }
    input_ = input;
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
        output_->print(name_ + "::output", true, input->get_batch_size());

    return output_;
}

void Softmax::bwd_initialize(Tensor<float> *target) {
    if (grad_input_ == nullptr || batch_size_ != target->get_batch_size()) {
        if (grad_input_ == nullptr)
            grad_input_ = new Tensor<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }
}

Tensor<float> *Softmax::backward(Tensor<float> *target) {
    bwd_initialize(target);
    // set grad_input_ as predict
    checkCudaErrors(cudaMemcpyAsync(
        grad_input_->get_device_ptr(),
        output_->get_device_ptr(),
        output_->buf_size(),
        cudaMemcpyDeviceToDevice));
    // set grad_input_ = predict - target
    checkCublasErrors(cublasSaxpy(
        cuda_->cublas(),
        target->len(),
        &cuda_->minus_one,
        target->get_device_ptr(),
        1,
        grad_input_->get_device_ptr(),
        1));

    // normalize the grad_output by the batch size
    int grad_output_size = target->get_batch_size() * target->get_channels() *
                           target->get_height() * target->get_width();
    float scale = 1.f / static_cast<float>(target->get_batch_size());
    checkCublasErrors(
        cublasSscal(cuda_->cublas(), grad_output_size, &scale, grad_input_->get_device_ptr(), 1));

    if (DEBUG_SOFTMAX > 1) {
        std::cout << name_ << "[BACKWARD]" << std::endl;
        input_->print(name_ + "::input", true);
        output_->print(name_ + "::predict", true);
        target->print(name_ + "::y", true, target->get_batch_size());
        grad_input_->print(name_ + "::dx", true, target->get_batch_size());
    }

    return grad_input_;
}
