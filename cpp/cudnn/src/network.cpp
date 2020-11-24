#include <network.h>

#include <iomanip>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

template <typename dtype> Network<dtype>::Network() {}

template <typename dtype> Network<dtype>::~Network() {
    for (auto layer : layers_)
        delete layer;

    if (cuda_ != nullptr)
        delete cuda_;
}

template <typename dtype> void Network<dtype>::add_layer(Layer<dtype> *layer) {
    layers_.push_back(layer);

    if (layers_.size() == 1)
        layers_.at(0)->set_gradient_stop();
}

template <typename dtype> Tensor<dtype> *Network<dtype>::forward(Tensor<dtype> *input) {
    output_ = input;

    nvtxRangePushA("Forward");
    for (auto layer : layers_) {
        if (DEBUG_FORWARD) {
            printf(
                "%s input squared before: %.20f\n",
                layer->get_name().c_str(),
                output_->get_magnitude_squared());
        }

        output_ = layer->forward(output_);

        if (DEBUG_FORWARD) {
            printf(
                "%s input squared before: %.20f\n",
                layer->get_name().c_str(),
                output_->get_magnitude_squared());
        }
    }
    nvtxRangePop();

    return output_;
}

template <typename dtype> void Network<dtype>::backward(Tensor<dtype> *target) {
    Tensor<dtype> *gradient = target;

    if (phase_ == inference)
        return;

    nvtxRangePushA("Backward");
    for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++) {
        if (DEBUG_BACKWARD) {
            printf(
                "%s grad squared before: %.20f\n",
                (*layer)->get_name().c_str(),
                gradient->get_magnitude_squared());
        }

        gradient = (*layer)->backward(gradient);

        if (DEBUG_BACKWARD) {
            printf(
                "%s grad squared after: %.20f\n",
                (*layer)->get_name().c_str(),
                gradient->get_magnitude_squared());
        }
    }
    nvtxRangePop();
}

// TODO: SGD and all that?
template <typename dtype> void Network<dtype>::update(double learning_rate) {
    if (phase_ == inference)
        return;

    if (DEBUG_UPDATE)
        std::cout << "start update.. lr = " << learning_rate << std::endl;

    nvtxRangePushA("Update");
    for (auto layer : layers_) {
        if (layer->weights_ == nullptr || layer->grad_weights_ == nullptr ||
            layer->biases_ == nullptr || layer->grad_biases_ == nullptr)
            continue;

        layer->update_weights_biases(learning_rate);
    }
    nvtxRangePop();
}

template <typename dtype> int Network<dtype>::write_file() {
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

template <typename dtype> int Network<dtype>::load_pretrain() {
    for (auto layer : layers_) {
        layer->set_load_pretrain();
    }

    return 0;
}

template <typename dtype> void Network<dtype>::cuda() {
    cuda_ = new CudaContext<dtype>();

    std::cout << ".. model Configuration .." << std::endl;
    for (auto layer : layers_) {
        layer->set_cuda_context(cuda_);
    }
}

template <typename dtype> void Network<dtype>::train() {
    phase_ = training;

    for (auto layer : layers_) {
        layer->unfreeze();
        layer->train();
        layer->zero_grad();
    }
}

template <typename dtype> void Network<dtype>::zero_grad() {
    for (auto layer : layers_) {
        layer->zero_grad();
    }
}


template <typename dtype> void Network<dtype>::eval() {
    phase_ = inference;

    // freeze all layers
    for (auto layer : layers_) {
        layer->freeze();
        layer->eval();
    }
}

template class Network<float>;

template class Network<double>;
