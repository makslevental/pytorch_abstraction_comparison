#include <network.h>

#include <iomanip>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

Network::Network() {
    // do nothing
}

Network::~Network() {
    // destroy network
    for (auto layer : layers_)
        delete layer;

    // terminate CUDA context
    if (cuda_ != nullptr)
        delete cuda_;
}

void Network::add_layer(Layer *layer) {
    layers_.push_back(layer);

    // tagging layer to stop gradient if it is the first layer
    if (layers_.size() == 1)
        layers_.at(0)->set_gradient_stop();
}

Tensor<float> *Network::forward(Tensor<float> *input) {
    output_ = input;

    nvtxRangePushA("Forward");
    for (auto layer : layers_) {
        if (DEBUG_FORWARD) {
            std::cout << "[[Forward ]][[ ";
            std::cout << std::setw(7) << layer->get_name() << " ]]\t(";
            std::cout << output_->shape() << std::endl;
        }

        layer->fwd_initialize(output_);
        output_ = layer->forward(output_);

#if (DEBUG_FORWARD)
        std::cout << "--> " << output_->shape() << std::endl;
        checkCudaErrors(cudaDeviceSynchronize());

#if (DEBUG_FORWARD > 1)
        output_->print("output", true);

        if (phase_ == inference)
            getchar();
#endif
#endif // DEBUG_FORWARD

        // TEST
        // checkCudaErrors(cudaDeviceSynchronize());
    }
    nvtxRangePop();

    return output_;
}

void Network::backward(Tensor<float> *target) {
    Tensor<float> *gradient = target;

    if (phase_ == inference)
        return;

    nvtxRangePushA("Backward");
    // back propagation.. update weights internally.....
    for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++) {
        // getting back propagation status with gradient size

        if (DEBUG_BACKWARD) {
            std::cout << "[[Backward]][[ " << std::setw(7) << (*layer)->get_name() << " ]]\t("
                      << gradient->get_batch_size() << ", " << gradient->get_channels() << ", "
                      << gradient->get_height() << ", " << gradient->get_width() << ")\t";
        }

        (*layer)->bwd_initialize(gradient);
        gradient = (*layer)->backward(gradient);

        if (DEBUG_BACKWARD) {
            // and the gradient result
            std::cout << "--> (" << gradient->get_batch_size() << ", " << gradient->get_channels()
                      << ", " << gradient->get_height() << ", " << gradient->get_width() << ")"
                      << std::endl;
            checkCudaErrors(cudaDeviceSynchronize());
        }

        if (DEBUG_BACKWARD > 1) {
            gradient->print((*layer)->get_name() + "::dx", true);
            //    getchar();
        }
    }
    nvtxRangePop();
}

void Network::update(float learning_rate) {
    if (phase_ == inference)
        return;

    if (DEBUG_UPDATE)
        std::cout << "Start update.. lr = " << learning_rate << std::endl;

    nvtxRangePushA("Update");
    for (auto layer : layers_) {
        // if no parameters, then pass
        if (layer->weights_ == nullptr || layer->grad_weights_ == nullptr ||
            layer->biases_ == nullptr || layer->grad_biases_ == nullptr)
            continue;

        layer->update_weights_biases(learning_rate);
    }
    nvtxRangePop();
}

int Network::write_file() {
    std::cout << ".. store weights to the storage .." << std::endl;
    for (auto layer : layers_) {
        int err = layer->save_parameter();

        if (err != 0) {
            std::cout << "-> error code: " << err << std::endl;
            exit(err);
        }
    }

    return 0;
}

int Network::load_pretrain() {
    for (auto layer : layers_) {
        layer->set_load_pretrain();
    }

    return 0;
}

// 1. initialize get_device_ptr resource container
// 2. register the resource container to all the layers
void Network::cuda() {
    cuda_ = new CudaContext();

    std::cout << ".. model Configuration .." << std::endl;
    for (auto layer : layers_) {
        std::cout << "CUDA: " << layer->get_name() << std::endl;
        layer->set_cuda_context(cuda_);
    }
}

//
void Network::train() {
    phase_ = training;

    // unfreeze all layers
    for (auto layer : layers_) {
        layer->unfreeze();
        layer->train();
    }
}

void Network::eval() {
    phase_ = inference;

    // freeze all layers
    for (auto layer : layers_) {
        layer->freeze();
        layer->eval();
    }
}

std::vector<Layer *> Network::layers() { return layers_; }

float Network::loss(Tensor<float> *target) {
    Layer *layer = layers_.back();
    return layer->get_loss(target);
}

int Network::get_accuracy(Tensor<float> *target) {
    Layer *layer = layers_.back();
    return layer->get_accuracy(target);
}
