#include <network.h>

#include <iomanip>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

template <typename dtype> Network<dtype>::Network() {
    // do nothing
}

template <typename dtype> Network<dtype>::~Network() {
    // destroy network
    for (auto layer : layers_)
        delete layer;

    // terminate CUDA context
    if (cuda_ != nullptr)
        delete cuda_;
}

template <typename dtype> void Network<dtype>::add_layer(Layer<dtype> *layer) {
    layers_.push_back(layer);

    // tagging layer to stop gradient if it is the first layer
    if (layers_.size() == 1)
        layers_.at(0)->set_gradient_stop();
}

template <typename dtype> Tensor<dtype> *Network<dtype>::forward(Tensor<dtype> *input) {
    output_ = input;

    nvtxRangePushA("Forward");
    for (auto layer : layers_) {
        if (DEBUG_FORWARD) {
            std::cout << "[[Forward ]][[ ";
            std::cout << std::setw(7) << layer->get_name() << " ]]\t(";
            std::cout << output_->shape() << std::endl;
            output_->print("input", true, output_->get_batch_size());
        }

        output_ = layer->forward(output_);

        if (DEBUG_FORWARD) {
            std::cout << "--> " << output_->shape() << std::endl;
            checkCudaErrors(cudaDeviceSynchronize());
        }
        if (DEBUG_FORWARD > 1) {
            output_->print("output", true, output_->get_batch_size());
            if (phase_ == inference)
                getchar();
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
    // back propagation.. update weights internally.....
    for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++) {
        // getting back propagation status with gradient size

        if (DEBUG_BACKWARD) {
            std::cout << "[[Backward]][[ " << std::setw(7) << (*layer)->get_name() << " ]]\t("
                      << gradient->get_batch_size() << ", " << gradient->get_channels() << ", "
                      << gradient->get_height() << ", " << gradient->get_width() << ")\t";
        }

        // TODO: stop storing things and pass them instead
        // TODO: figure out why the beginning of the epoch accuracy is low
        gradient = (*layer)->backward(gradient);

        // TODO change debugging to flags
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

// TODO: SGD and all that?
template <typename dtype> void Network<dtype>::update(double learning_rate) {
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
        std::cout << "CUDA: " << layer->get_name() << std::endl;
        layer->set_cuda_context(cuda_);
    }
}

//
template <typename dtype> void Network<dtype>::train() {
    phase_ = training;

    // unfreeze all layers
    for (auto layer : layers_) {
        layer->unfreeze();
        layer->train();
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
