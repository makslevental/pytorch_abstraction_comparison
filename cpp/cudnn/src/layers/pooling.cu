//
// Created by Maksim Levental on 10/29/20.
//

#include <layers/pooling.cuh>

template <typename dtype> Pooling<dtype>::Pooling(
    std::string name, int kernel_size, int stride, int padding, cudnnPoolingMode_t mode)
    : kernel_size_(kernel_size), padding_(padding), stride_(stride), mode_(mode) {
    this->name_ = std::move(name);

    cudnnCreatePoolingDescriptor(&pool_desc_);
    cudnnSetPooling2dDescriptor(
        pool_desc_,
        mode_,
        CUDNN_PROPAGATE_NAN,
        kernel_size_,
        kernel_size_,
        padding_,
        padding_,
        stride_,
        stride_);
}

template <typename dtype> Pooling<dtype>::~Pooling() { cudnnDestroyPoolingDescriptor(pool_desc_); }

template <typename dtype> void Pooling<dtype>::fwd_initialize(Tensor<dtype> *input) {
    if (this->input_desc_ == nullptr || this->batch_size_ != input->get_batch_size()) {
        this->input_size_ = input->size();
        // resource initialize
        this->input_desc_ = input->tensor_descriptor();
        this->batch_size_ = input->get_batch_size();

        // setting output
        cudnnGetPooling2dForwardOutputDim(
            pool_desc_,
            this->input_desc_,
            &output_size_[0],
            &output_size_[1],
            &output_size_[2],
            &output_size_[3]);
        if (this->output_ == nullptr)
            this->output_ = new Tensor<dtype>(output_size_);
        else
            this->output_->reset(output_size_);

        this->output_desc_ = this->output_->tensor_descriptor();
    }
}

template <typename dtype> Tensor<dtype> *Pooling<dtype>::forward(Tensor<dtype> *input) {
    fwd_initialize(input);
    this->input_ = input;
    cudnnPoolingForward(
        this->cuda_->cudnn(),
        pool_desc_,
        &this->cuda_->one,
        this->input_desc_,
        input->get_device_ptr(),
        &this->cuda_->zero,
        this->output_desc_,
        this->output_->get_device_ptr());

    return this->output_;
}

template <typename dtype> Tensor<dtype> *Pooling<dtype>::backward(Tensor<dtype> *grad_output) {
    this->bwd_initialize(grad_output);
    checkCudnnErrors(cudnnPoolingBackward(
        this->cuda_->cudnn(),
        pool_desc_,
        &this->cuda_->one,
        this->output_desc_,
        this->output_->get_device_ptr(),
        this->output_desc_,
        grad_output->get_device_ptr(),
        this->input_desc_,
        this->input_->get_device_ptr(),
        &this->cuda_->zero,
        this->input_desc_,
        this->grad_of_input_->get_device_ptr()));

    return this->grad_of_input_;
}

template class Pooling<float>;
template class Pooling<double>;
