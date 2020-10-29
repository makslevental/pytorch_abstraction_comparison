//
// Created by Maksim Levental on 10/29/20.
//

#include <cassert>
#include <softmax.cuh>

Softmax::Softmax(std::string name) { name_ = std::move(name); }

void Softmax::fwd_initialize(Tensor<float> *input) {
    if (input_ == nullptr || batch_size_ != input->get_batch_size()) {
        input_ = input;
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
    if (DEBUG_SOFTMAX & 0x01) {
        std::cout << name_ << "[FORWARD]" << std::endl;
        input_->print(name_ + "::input", true, input->get_batch_size());
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

    if (DEBUG_SOFTMAX & 0x01)
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

    if (DEBUG_SOFTMAX & 0x02) {
        std::cout << name_ << "[BACKWARD]" << std::endl;
        input_->print(name_ + "::input", true);
        output_->print(name_ + "::predict", true);
        target->print(name_ + "::y", true, target->get_batch_size());
        grad_input_->print(name_ + "::dx", true, target->get_batch_size());
    }

    return grad_input_;
}

float Softmax::get_loss(Tensor<float> *target) { return loss_.loss(output_, target); }

int Softmax::get_accuracy(Tensor<float> *target) {
    int batch_size = output_->get_batch_size();
    int output_size = output_->size();

    assert(batch_size == target->get_batch_size());
    assert(output_size == target->size());

    float *h_output, *h_target;
    int idx_output, idx_target;
    int hit_count = 0;

    // get predicts and targets
    h_output = output_->to(host);
    h_target = target->to(host);

    // idx_output = idx_target = 0;
    for (int b = 0; b < batch_size; b++) {
        idx_output = 0;
        idx_target = 0;

        for (int i = 1; i < 10; i++) {
            if (h_output[b * output_size + i] > h_output[b * output_size + idx_output])
                idx_output = i;
            if (h_target[b * output_size + i] > h_target[b * output_size + idx_target])
                idx_target = i;
        }

        if (idx_output == idx_target)
            hit_count++;
    }

    return hit_count;
}
