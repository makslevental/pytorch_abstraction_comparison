//
// Created by Maksim Levental on 10/29/20.
//

#include <layers/dense.cuh>

/****************************************************************
 * Dense<dtype> Layer                                                  *
 ****************************************************************/

template <typename dtype> Dense<dtype>::Dense(std::string name, int output_size) {
    this->name_ = std::move(name);
    output_size_ = output_size;
}

template <typename dtype> Dense<dtype>::~Dense<dtype>() {
    if (d_one_vec != nullptr) {
        cudaFree(d_one_vec);
        d_one_vec = nullptr;
    }
}

template <typename dtype> __global__ void init_one_vec(dtype *d_one_vec, size_t length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= length)
        return;

    d_one_vec[i] = 1.f;
}

template <typename dtype> void Dense<dtype>::fwd_initialize(Tensor<dtype> *input) {
    // initialize weights and biases
    if (this->weights_ == nullptr) {
        this->input_size_ = input->size();
        this->weights_ = new Tensor<dtype>(1, 1, this->input_size_, output_size_);
        this->biases_ = new Tensor<dtype>(1, 1, output_size_);
    }

    if (this->input_desc_ == nullptr || this->batch_size_ != input->get_batch_size()) {
        this->input_size_ = input->size();
        this->input_desc_ = input->tensor_descriptor();
        this->batch_size_ = input->get_batch_size();

        if (this->output_ == nullptr)
            this->output_ = new Tensor<dtype>(this->batch_size_, output_size_);
        else
            this->output_->reset(this->batch_size_, output_size_);

        this->output_->tensor_descriptor();

        if (d_one_vec != nullptr)
            cudaFree(d_one_vec);
        checkCudaErrors(cudaMalloc((void **)&d_one_vec, sizeof(dtype) * this->batch_size_));

        init_one_vec<<<(this->batch_size_ + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D>>>(
            d_one_vec, this->batch_size_);

        if (this->load_pretrain_ && !this->freeze_) {
            if (this->load_parameter()) {
                std::cout << "error occurred.." << std::endl;
                exit(-1);
            }
        } else if (!this->freeze_) {
            this->init_weight_bias();
//            this->weights_->print("weights", true);
//            this->biases_->print("biases", true);
        }
    }
}

template <typename dtype> Tensor<dtype> *Dense<dtype>::forward(Tensor<dtype> *input) {
    // output = weights^T * input (without biases)
    fwd_initialize(input);
    this->input_ = input;
    if constexpr (std::is_same<dtype, float>{}) {
        checkCublasErrors(cublasSgemm(
            this->cuda_->cublas(),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            output_size_,
            this->batch_size_,
            this->input_size_,
            &this->cuda_->one,
            this->weights_->get_device_ptr(),
            this->input_size_,
            input->get_device_ptr(),
            this->input_size_,
            &this->cuda_->zero,
            this->output_->get_device_ptr(),
            output_size_));

        // output += biases * d_one_vec^T
        checkCublasErrors(cublasSgemm(
            this->cuda_->cublas(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            output_size_,
            this->batch_size_,
            1,
            &this->cuda_->one,
            this->biases_->get_device_ptr(),
            output_size_,
            d_one_vec,
            1,
            &this->cuda_->one,
            this->output_->get_device_ptr(),
            output_size_));
    } else if constexpr (std::is_same<dtype, double>{}) {
        checkCublasErrors(cublasDgemm(
            this->cuda_->cublas(),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            output_size_,
            this->batch_size_,
            this->input_size_,
            &this->cuda_->one,
            this->weights_->get_device_ptr(),
            this->input_size_,
            input->get_device_ptr(),
            this->input_size_,
            &this->cuda_->zero,
            this->output_->get_device_ptr(),
            output_size_));

        // output += biases * d_one_vec^T
        checkCublasErrors(cublasDgemm(
            this->cuda_->cublas(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            output_size_,
            this->batch_size_,
            1,
            &this->cuda_->one,
            this->biases_->get_device_ptr(),
            output_size_,
            d_one_vec,
            1,
            &this->cuda_->one,
            this->output_->get_device_ptr(),
            output_size_));
    }

    if (DEBUG_DENSE) {
        input->print(this->name_ + "::input", true);
        this->weights_->print(this->name_ + "::weight", true);
        this->biases_->print(this->name_ + "::bias", true);
        this->output_->print(this->name_ + "::output", true);
    }

    return this->output_;
}

template <typename dtype> void Dense<dtype>::bwd_initialize(Tensor<dtype> *grad_output) {
    if (this->grad_weights_ == nullptr) {
        this->grad_weights_ = new Tensor<dtype>(this->weights_->shape());
        this->grad_biases_ = new Tensor<dtype>(this->biases_->shape());
    }
    Layer<dtype>::bwd_initialize(grad_output);
}

template <typename dtype> Tensor<dtype> *Dense<dtype>::backward(Tensor<dtype> *grad_output) {
    bwd_initialize(grad_output);
    // db = (dy) * d_one_vec
    if constexpr (std::is_same<dtype, float>{}) {
        checkCublasErrors(cublasSgemv(
            this->cuda_->cublas(),
            CUBLAS_OP_N,
            output_size_,
            this->batch_size_,
            &this->cuda_->one,
            this->grad_of_output_->get_device_ptr(),
            output_size_,
            d_one_vec,
            1,
            &this->cuda_->zero,
            this->grad_biases_->get_device_ptr(),
            1));

        // dw = x * (dy)^T
        checkCublasErrors(cublasSgemm(
            this->cuda_->cublas(),
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            this->input_size_,
            output_size_,
            this->batch_size_,
            &this->cuda_->one,
            this->input_->get_device_ptr(),
            this->input_size_,
            this->grad_of_output_->get_device_ptr(),
            output_size_,
            &this->cuda_->zero,
            this->grad_weights_->get_device_ptr(),
            this->input_size_));

    } else if constexpr (std::is_same<dtype, double>{}) {
        checkCublasErrors(cublasDgemv(
            this->cuda_->cublas(),
            CUBLAS_OP_N,
            output_size_,
            this->batch_size_,
            &this->cuda_->one,
            this->grad_of_output_->get_device_ptr(),
            output_size_,
            d_one_vec,
            1,
            &this->cuda_->zero,
            this->grad_biases_->get_device_ptr(),
            1));

        // dw = x * (dy)^T
        checkCublasErrors(cublasDgemm(
            this->cuda_->cublas(),
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            this->input_size_,
            output_size_,
            this->batch_size_,
            &this->cuda_->one,
            this->input_->get_device_ptr(),
            this->input_size_,
            this->grad_of_output_->get_device_ptr(),
            output_size_,
            &this->cuda_->zero,
            this->grad_weights_->get_device_ptr(),
            this->input_size_));
    }

    // dx = W * dy
    if (!this->gradient_stop_)
        if constexpr (std::is_same<dtype, float>{}) {
            checkCublasErrors(cublasSgemm(
                this->cuda_->cublas(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                this->input_size_,
                this->batch_size_,
                output_size_,
                &this->cuda_->one,
                this->weights_->get_device_ptr(),
                this->input_size_,
                this->grad_of_output_->get_device_ptr(),
                output_size_,
                &this->cuda_->zero,
                this->grad_of_input_->get_device_ptr(),
                this->input_size_));

        } else if constexpr (std::is_same<dtype, double>{}) {
            checkCublasErrors(cublasDgemm(
                this->cuda_->cublas(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                this->input_size_,
                this->batch_size_,
                output_size_,
                &this->cuda_->one,
                this->weights_->get_device_ptr(),
                this->input_size_,
                this->grad_of_output_->get_device_ptr(),
                output_size_,
                &this->cuda_->zero,
                this->grad_of_input_->get_device_ptr(),
                this->input_size_));
        }

    if (DEBUG_DENSE & 0x02) {
        std::cout << this->name_ << "[BACKWARD]" << std::endl;
        grad_output->print(this->name_ + "::gradients", true, grad_output->get_batch_size());
        this->grad_weights_->print(this->name_ + "::gfilter", true);
        this->grad_biases_->print(this->name_ + "::gbias", true);
        if (!this->gradient_stop_)
            this->grad_of_input_->print(this->name_ + "::gdata", true);
    }

    return this->grad_of_input_;
}

template <typename dtype> std::tuple<int, int> Dense<dtype>::calculate_fan_in_and_fan_out() {
    auto num_input_fmaps = this->input_->get_channels();
    auto num_output_fmaps = this->output_->get_channels();
    auto receptive_field_size = 1;
    return std::make_tuple(
        num_input_fmaps * receptive_field_size, num_output_fmaps * receptive_field_size);
}

template class Dense<float>;
template class Dense<double>;
