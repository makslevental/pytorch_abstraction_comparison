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

enum Nonlinearity {
    LINEAR,
    CONV1D,
    CONV2D,
    CONV3D,
    CONV_TRANSPOSE1D,
    CONV_TRANSPOSE2D,
    CONV_TRANSPOSE3D,
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU
};

double calculate_gain(Nonlinearity nonlinearity, double param = -1) {
    switch (nonlinearity) {
    case LINEAR:
    case CONV1D:
    case CONV2D:
    case CONV3D:
    case CONV_TRANSPOSE1D:
    case CONV_TRANSPOSE2D:
    case CONV_TRANSPOSE3D:
    case SIGMOID:
        return 1.0;
    case TANH:
        return 5.0 / 3;
    case RELU:
        return sqrt(2.0);
    case LEAKY_RELU:
        double negative_slope;
        if (param == -1) {
            negative_slope = 0.01;
        } else {
            negative_slope = param;
        }
        return sqrt(2.0 / (1 + pow(negative_slope, 2)));
    }
    std::cout << "couldn't compute gain\n";
    exit(EXIT_FAILURE);
}

// wtf this is necessary here but not in libtorch nor pytorch
template <typename dtype> void Layer<dtype>::init_weight_bias(unsigned int seed) {
    if (weights_ == nullptr || biases_ == nullptr)
        return;
    // Create random network
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

    auto [fan_in, fan_out] = calculate_fan_in_and_fan_out();
    auto gain = calculate_gain(LEAKY_RELU, sqrt(5.0));
    auto std = gain / fan_in;
    auto bound = sqrt(3.0) * std;
//    printf("%s weights %d %f %f %f\n", this->name_.c_str(), fan_in, gain, std, bound);
    std::uniform_real_distribution<> dis1(-bound, bound);
    for (int i = 0; i < weights_->len(); i++)
        weights_->get_host_ptr()[i] = static_cast<double>(dis1(gen));

    bound = 1 / sqrt(fan_in);
//    printf("%s biases %f\n", this->name_.c_str(), bound);
    std::uniform_real_distribution<> dis2(-bound, bound);
    for (int i = 0; i < biases_->len(); i++)
        biases_->get_host_ptr()[i] = static_cast<double>(dis2(gen));

    weights_->to(cuda);
    biases_->to(cuda);
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

template <typename dtype> void Layer<dtype>::zero_grad() {
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

template <typename dtype> std::tuple<int, int> Layer<dtype>::calculate_fan_in_and_fan_out() {
    return std::tuple<int, int>{1, 1};
}

template class Layer<float>;
template class Layer<double>;
