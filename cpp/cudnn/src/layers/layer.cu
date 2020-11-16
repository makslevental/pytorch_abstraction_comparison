#include <layers/layer.h>

#include <random>

#include <algorithm>
#include <cassert>
#include <cmath>

#include <fstream>
#include <iostream>
#include <sstream>

/****************************************************************
 * Layer definition                                             *
 ****************************************************************/
template <typename dtype> Layer<dtype>::Layer() { /* do nothing */
}

template <typename dtype> Layer<dtype>::~Layer() {
    if (DEBUG_FORWARD > 0 || DEBUG_BACKWARD > 0)
        std::cout << "Destroy Layer: " << name_ << std::endl;

    if (output_ != nullptr) {
        delete output_;
        output_ = nullptr;
    }
    if (grad_of_input_ != nullptr) {
        delete grad_of_input_;
        grad_of_input_ = nullptr;
    }

    if (weights_ != nullptr) {
        delete weights_;
        weights_ = nullptr;
    }
    if (biases_ != nullptr) {
        delete biases_;
        biases_ = nullptr;
    }
    if (grad_weights_ != nullptr) {
        delete grad_weights_;
        grad_weights_ = nullptr;
    }
    if (grad_biases_ != nullptr) {
        delete grad_biases_;
        grad_biases_ = nullptr;
    }
}

template <typename dtype> void Layer<dtype>::init_weight_bias(unsigned int seed) {
    if (weights_ == nullptr || biases_ == nullptr)
        return;
    // Create random network
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

    // He uniform distribution
    // TODO: initialization Xi
    double range = sqrt(6.f / input_size_); // He's initialization
    std::uniform_real_distribution<> dis(-range, range);

    for (int i = 0; i < weights_->len(); i++)
        weights_->get_host_ptr()[i] = static_cast<double>(dis(gen));
    for (int i = 0; i < biases_->len(); i++)
        biases_->get_host_ptr()[i] = 0.f;

    // copy initialized value to the device
    weights_->to(DeviceType::cuda);
    biases_->to(DeviceType::cuda);

    std::cout << ".. initialized " << name_ << " layer .." << std::endl;
}

template <typename dtype> void Layer<dtype>::update_weights_biases(dtype learning_rate) {
    dtype eps = -1.f * learning_rate;
    if (weights_ != nullptr && grad_weights_ != nullptr) {
        if (DEBUG_UPDATE) {
            weights_->print(name_ + "::weights (before update)", true);
            grad_weights_->print(name_ + "::gweights", true);
        }

        // w = w + eps * dw
        if constexpr (std::is_same<dtype, float>{}) {
            checkCublasErrors(cublasSaxpy(
                cuda_->cublas(),
                weights_->len(),
                &eps,
                grad_weights_->get_device_ptr(),
                1,
                weights_->get_device_ptr(),
                1));
        } else if constexpr (std::is_same<dtype, double>{}) {
            checkCublasErrors(cublasDaxpy(
                cuda_->cublas(),
                weights_->len(),
                &eps,
                grad_weights_->get_device_ptr(),
                1,
                weights_->get_device_ptr(),
                1));
        }

        if (DEBUG_UPDATE)
            weights_->print(name_ + "weights (after update)", true);
    }

    if (biases_ != nullptr && grad_biases_ != nullptr) {
        if (DEBUG_UPDATE) {
            biases_->print(name_ + "biases (before update)", true);
            grad_biases_->print(name_ + "gbiases", true);
        }

        // b = b + eps * db
        if constexpr (std::is_same<dtype, float>{}) {
            checkCublasErrors(cublasSaxpy(
                cuda_->cublas(),
                biases_->len(),
                &eps,
                grad_biases_->get_device_ptr(),
                1,
                biases_->get_device_ptr(),
                1));
        } else if constexpr (std::is_same<dtype, double>{}) {
            checkCublasErrors(cublasDaxpy(
                cuda_->cublas(),
                biases_->len(),
                &eps,
                grad_biases_->get_device_ptr(),
                1,
                biases_->get_device_ptr(),
                1));
        }

        if (DEBUG_UPDATE)
            biases_->print(name_ + "biases (after update)", true);
    }
}

template <typename dtype> void Layer<dtype>::fwd_initialize(Tensor<dtype> *input) {
    if (input_desc_ == nullptr || batch_size_ != input->get_batch_size()) {
        //        input_ = input;
        input_size_ = input->size();
        input_desc_ = input->tensor_descriptor();
        batch_size_ = input->get_batch_size();

        if (output_ == nullptr)
            output_ = new Tensor<dtype>(input->shape());
        else
            output_->reset(input->shape());

        output_desc_ = output_->tensor_descriptor();
    }
}

template <typename dtype> void Layer<dtype>::bwd_initialize(Tensor<dtype> *grad_output) {
    if (grad_of_input_ == nullptr || batch_size_ != grad_output->get_batch_size()) {
        grad_of_output_ = grad_output;

        if (grad_of_input_ == nullptr)
            grad_of_input_ = new Tensor<dtype>(input_->shape());
        else
            grad_of_input_->reset(input_->shape());
    }
}

template <typename dtype> int Layer<dtype>::load_parameter() {
    std::stringstream filename_weights, filename_biases;

    // load weights and biases pretrained parameters
    filename_weights << name_ << ".bin";
    if (weights_->file_read(filename_weights.str()))
        return -1;

    filename_biases << name_ << ".bias.bin";
    if (biases_->file_read(filename_biases.str()))
        return -2;

    std::cout << ".. loaded " << name_ << " pretrain parameter.." << std::endl;

    return 0;
}

template <typename dtype> int Layer<dtype>::save_parameter() {
    std::stringstream filename_weights, filename_biases;

    std::cout << ".. saving " << name_ << " parameter ..";

    // Write weights file
    if (weights_) {
        filename_weights << name_ << ".bin";
        if (weights_->file_write(filename_weights.str()))
            return -1;
    }

    // Write bias file
    if (biases_) {
        filename_biases << name_ << ".bias.bin";
        if (biases_->file_write(filename_biases.str()))
            return -2;
    }

    std::cout << " done .." << std::endl;

    return 0;
}

template <typename dtype> void Layer<dtype>::zero_out() {
    if (input_) {
        input_->zero_out();
    }
    if (output_) {
        output_->zero_out();
    }
    if (grad_of_input_) {
        grad_of_input_->zero_out();
    }
    if (grad_of_output_) {
        grad_of_output_->zero_out();
    }
    if (grad_biases_) {
        grad_biases_->zero_out();
    }
    if (grad_weights_) {
        grad_weights_->zero_out();
    }
}

template class Layer<float>;
template class Layer<double>;
