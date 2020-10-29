//
// Created by Maksim Levental on 10/29/20.
//

#include <dense.cuh>

/****************************************************************
 * Dense Layer                                                  *
 ****************************************************************/

Dense::Dense(std::string name, int output_size) {
    name_ = std::move(name);
    output_size_ = output_size;
}

Dense::~Dense() {
    if (d_one_vec != nullptr) {
        cudaFree(d_one_vec);
        d_one_vec = nullptr;
    }
}

__global__ void init_one_vec(float *d_one_vec, size_t length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= length)
        return;

    d_one_vec[i] = 1.f;
}

void Dense::fwd_initialize(Tensor<float> *input) {
    // initialize weights and biases
    if (weights_ == nullptr) {
        // setup parameter size information
        input_size_ = input->get_channels() * input->get_height() * input->get_height();

        // initialize weight, bias, and output
        weights_ = new Tensor<float>(1, 1, input_size_, output_size_);
        biases_ = new Tensor<float>(1, 1, output_size_);
    }

    // initilaize input and output
    if (input_ == nullptr || batch_size_ != input->get_batch_size()) {
        input_ = input;
        batch_size_ = input->get_batch_size();

        if (output_ == nullptr)
            output_ = new Tensor<float>(batch_size_, output_size_);
        else
            output_->reset(batch_size_, output_size_);

        output_->tensor_descriptor();

        if (d_one_vec != nullptr)
            cudaFree(d_one_vec);
        checkCudaErrors(cudaMalloc((void **)&d_one_vec, sizeof(float) * batch_size_));
        init_one_vec<<<(batch_size_ + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D>>>(
            d_one_vec, batch_size_);

        // initialize weights and biases
        if (load_pretrain_ && !freeze_) {
            if (load_parameter()) {
                std::cout << "error occurred.." << std::endl;
                exit(-1);
            }
        } else if (!freeze_) {
            init_weight_bias();
        } else {
            /* do nothing */
        }
    }
}

Tensor<float> *Dense::forward(Tensor<float> *input) {
    // output = weights^T * input (without biases)
    checkCublasErrors(cublasSgemm(
        cuda_->cublas(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        output_size_,
        batch_size_,
        input_size_,
        &cuda_->one,
        weights_->get_device_ptr(),
        input_size_,
        input_->get_device_ptr(),
        input_size_,
        &cuda_->zero,
        output_->get_device_ptr(),
        output_size_));

    // output += biases * d_one_vec^T
    checkCublasErrors(cublasSgemm(
        cuda_->cublas(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        output_size_,
        batch_size_,
        1,
        &cuda_->one,
        biases_->get_device_ptr(),
        output_size_,
        d_one_vec,
        1,
        &cuda_->one,
        output_->get_device_ptr(),
        output_size_));

#if (DEBUG_DENSE & 0x01)
    input_->print(name_ + "::input", true);
    weights_->print(name_ + "::weight", true);
    biases_->print(name_ + "::bias", true);
    output_->print(name_ + "::output", true);
#endif // DEBUG_DENSE

    return output_;
}

void Dense::bwd_initialize(Tensor<float> *grad_output) {
    if (grad_weights_ == nullptr) {
        grad_weights_ = new Tensor<float>(weights_->shape());
        grad_biases_ = new Tensor<float>(biases_->shape());
    }
    Layer::bwd_initialize(grad_output);
}

Tensor<float> *Dense::backward(Tensor<float> *grad_output) {
    // db = (dy) * d_one_vec
    cublasSgemv(
        cuda_->cublas(),
        CUBLAS_OP_N,
        output_size_,
        batch_size_,
        &cuda_->one,
        grad_output_->get_device_ptr(),
        output_size_,
        d_one_vec,
        1,
        &cuda_->zero,
        grad_biases_->get_device_ptr(),
        1);

    // dw = x * (dy)^T
    cublasSgemm(
        cuda_->cublas(),
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        input_size_,
        output_size_,
        batch_size_,
        &cuda_->one,
        input_->get_device_ptr(),
        input_size_,
        grad_output_->get_device_ptr(),
        output_size_,
        &cuda_->zero,
        grad_weights_->get_device_ptr(),
        input_size_);

    // dx = W * dy
    if (!gradient_stop_)
        cublasSgemm(
            cuda_->cublas(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            input_size_,
            batch_size_,
            output_size_,
            &cuda_->one,
            weights_->get_device_ptr(),
            input_size_,
            grad_output_->get_device_ptr(),
            output_size_,
            &cuda_->zero,
            grad_input_->get_device_ptr(),
            input_size_);

#if (DEBUG_DENSE & 0x02)
    std::cout << name_ << "[BACKWARD]" << std::endl;
    grad_output->print(name_ + "::gradients", true, grad_output->n());
    grad_weights_->print(name_ + "::gfilter", true);
    grad_biases_->print(name_ + "::gbias", true);
    if (!gradient_stop_)
        grad_input_->print(name_ + "::gdata", true);
#endif // DEBUG_DENSE

    return grad_input_;
}
