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

Tensor<double> *Network::forward(Tensor<double> *input, int DEBUG) {
    output_ = input;

    nvtxRangePushA("Forward");
    std::stringstream ss;
    for (auto layer : layers_) {
        if (DEBUG) {
            std::cout << "[[Forward ]][[ ";
            ss.str("");
            ss << std::setw(7) << layer->get_name() << " layer input";
            output_->print(ss.str(), true, output_->get_batch_size());
        }

        output_ = layer->forward(output_);

        if (DEBUG) {
            checkCudaErrors(cudaDeviceSynchronize());
        }
        if (DEBUG > 1) {
            ss.str("");
            ss << std::setw(7) << layer->get_name() << " layer output";
            output_->print(ss.str(), true, output_->get_batch_size());
        }
    }
    nvtxRangePop();

    return output_;
}

void Network::backward(Tensor<double> *gradient) {
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
void Network::update(double learning_rate) {
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
        layer->zero_out();
    }
}

void Network::eval() {
    phase_ = inference;

    // freeze all layers
    for (auto layer : layers_) {
        layer->freeze();
        layer->eval();
        layer->zero_out();
    }
}

Tensor<double> *Network::forward(Tensor<double> *input) { return forward(input, 0); }
CudaContext *Network::get_cuda_context() const { return cuda_; }
