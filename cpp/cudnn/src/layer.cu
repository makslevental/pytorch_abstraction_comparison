#include <layer.h>

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
Layer::Layer() { /* do nothing */
}

Layer::~Layer() {
    if (DEBUG_FORWARD > 0 || DEBUG_BACKWARD > 0)
        std::cout << "Destroy Layer: " << name_ << std::endl;

    if (output_ != nullptr) {
        delete output_;
        output_ = nullptr;
    }
    if (grad_input_ != nullptr) {
        delete grad_input_;
        grad_input_ = nullptr;
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

void Layer::init_weight_bias(unsigned int seed) {
    checkCudaErrors(cudaDeviceSynchronize());

    if (weights_ == nullptr || biases_ == nullptr)
        return;
    PRINT("init weights biases");
    // Create random network
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

    // He uniform distribution
    // TODO: initialization Xi
    float range = sqrt(6.f / input_size_); // He's initialization
    std::uniform_real_distribution<> dis(-range, range);

    for (int i = 0; i < weights_->len(); i++)
        weights_->get_host_ptr()[i] = static_cast<float>(dis(gen));
    for (int i = 0; i < biases_->len(); i++)
        biases_->get_host_ptr()[i] = 0.f;

    // copy initialized value to the device
    weights_->to(DeviceType::cuda);
    biases_->to(DeviceType::cuda);

    std::cout << ".. initialized " << name_ << " layer .." << std::endl;
}

void Layer::update_weights_biases(float learning_rate) {
    float eps = -1.f * learning_rate;
    if (weights_ != nullptr && grad_weights_ != nullptr) {
        if (DEBUG_UPDATE) {
            weights_->print(name_ + "::weights (before update)", true);
            grad_weights_->print(name_ + "::gweights", true);
        }

        // w = w + eps * dw
        checkCublasErrors(cublasSaxpy(
            cuda_->cublas(),
            weights_->len(),
            &eps,
            grad_weights_->get_device_ptr(),
            1,
            weights_->get_device_ptr(),
            1));

        if (DEBUG_UPDATE)
            weights_->print(name_ + "weights (after update)", true);
    }

    if (biases_ != nullptr && grad_biases_ != nullptr) {
        if (DEBUG_UPDATE) {
            biases_->print(name_ + "biases (before update)", true);
            grad_biases_->print(name_ + "gbiases", true);
        }

        // b = b + eps * db
        checkCublasErrors(cublasSaxpy(
            cuda_->cublas(),
            biases_->len(),
            &eps,
            grad_biases_->get_device_ptr(),
            1,
            biases_->get_device_ptr(),
            1));

        if (DEBUG_UPDATE)
            biases_->print(name_ + "biases (after update)", true);
    }
}

void Layer::fwd_initialize(Tensor<float> *input) {
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

void Layer::bwd_initialize(Tensor<float> *grad_output) {
    if (grad_input_ == nullptr || batch_size_ != grad_output->get_batch_size()) {
        grad_output_ = grad_output;

        if (grad_input_ == nullptr)
            grad_input_ = new Tensor<float>(input_->shape());
        else
            grad_input_->reset(input_->shape());
    }
}

int Layer::load_parameter() {
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

int Layer::save_parameter() {
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
