//
// Created by Maksim Levental on 10/29/20.
//

#include <cassert>
#include <layers/softmax.cuh>

template <typename dtype> Softmax<dtype>::Softmax(std::string name) {
    this->name_ = std::move(name);
}

template <typename dtype> void Softmax<dtype>::fwd_initialize(Tensor<dtype> *input) {
    if (this->input_desc_ == nullptr || this->batch_size_ != input->get_batch_size()) {
        this->input_desc_ = input->tensor_descriptor();
        this->batch_size_ = input->get_batch_size();

        if (this->output_ == nullptr)
            this->output_ = new Tensor<dtype>(input->shape());
        else
            this->output_->reset(input->shape());

        this->output_desc_ = this->output_->tensor_descriptor();
    }
}

template <typename dtype> Tensor<dtype> *Softmax<dtype>::forward(Tensor<dtype> *input) {
    fwd_initialize(input);
    if (DEBUG_SOFTMAX) {
        std::cout << this->name_ << "[FORWARD]" << std::endl;
        this->input_->print(this->name_ + "::input", true, input->get_batch_size());
    }
    this->input_ = input;
    checkCudnnErrors(cudnnSoftmaxForward(
        this->cuda_->cudnn(),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &this->cuda_->one,
        this->input_desc_,
        input->get_device_ptr(),
        &this->cuda_->zero,
        this->output_desc_,
        this->output_->get_device_ptr()));

    if (DEBUG_SOFTMAX)
        this->output_->print(this->name_ + "::output", true, input->get_batch_size());

    return this->output_;
}

template <typename dtype> void Softmax<dtype>::bwd_initialize(Tensor<dtype> *target) {
    if (this->grad_input_ == nullptr || this->batch_size_ != target->get_batch_size()) {
        if (this->grad_input_ == nullptr)
            this->grad_input_ = new Tensor<dtype>(this->input_->shape());
        else
            this->grad_input_->reset(this->input_->shape());
    }
}

template <typename dtype> Tensor<dtype> *Softmax<dtype>::backward(Tensor<dtype> *target) {
    bwd_initialize(target);
    // set grad_input_ as predict
    checkCudaErrors(cudaMemcpyAsync(
        this->grad_input_->get_device_ptr(),
        this->output_->get_device_ptr(),
        this->output_->buf_size(),
        cudaMemcpyDeviceToDevice));
    // set grad_input_ = predict - target
    if constexpr (std::is_same<dtype, float>{}) {
        checkCublasErrors(cublasSaxpy(
            this->cuda_->cublas(),
            target->len(),
            &this->cuda_->minus_one,
            target->get_device_ptr(),
            1,
            this->grad_input_->get_device_ptr(),
            1));
    } else if constexpr (std::is_same<dtype, double>{}) {
        checkCublasErrors(cublasDaxpy(
            this->cuda_->cublas(),
            target->len(),
            &this->cuda_->minus_one,
            target->get_device_ptr(),
            1,
            this->grad_input_->get_device_ptr(),
            1));
    }


    // normalize the grad_output by the batch size
    int grad_output_size = target->get_batch_size() * target->get_channels() *
                           target->get_height() * target->get_width();
    dtype scale = 1.f / static_cast<dtype>(target->get_batch_size());
    if constexpr (std::is_same<dtype, float>{}) {
        checkCublasErrors(cublasSscal(
            this->cuda_->cublas(),
            grad_output_size,
            &scale,
            this->grad_input_->get_device_ptr(),
            1));
    } else if constexpr (std::is_same<dtype, double>{}) {
        checkCublasErrors(cublasDscal(
            this->cuda_->cublas(),
            grad_output_size,
            &scale,
            this->grad_input_->get_device_ptr(),
            1));
    }

    if (DEBUG_SOFTMAX > 1) {
        std::cout << this->name_ << "[BACKWARD]" << std::endl;
        this->input_->print(this->name_ + "::input", true);
        this->output_->print(this->name_ + "::predict", true);
        target->print(this->name_ + "::y", true, target->get_batch_size());
        this->grad_input_->print(this->name_ + "::dx", true, target->get_batch_size());
    }

    return this->grad_input_;
}

template class Softmax<float>;

template class Softmax<double>;
