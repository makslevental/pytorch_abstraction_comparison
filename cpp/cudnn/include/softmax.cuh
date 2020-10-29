//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_SOFTMAX_CUH
#define PROJECTNAME_SOFTMAX_CUH

#include "layer.h"

class Softmax : public Layer {
public:
    explicit Softmax(std::string name);
    ~Softmax() override = default;

    Tensor<float> *forward(Tensor<float> *input) override;
    Tensor<float> *backward(Tensor<float> *grad_input) override;

    float get_loss(Tensor<float> *target) override;
    int get_accuracy(Tensor<float> *target) override;

protected:
    void fwd_initialize(Tensor<float> *input) override;
    void bwd_initialize(Tensor<float> *grad_output) override;

    CrossEntropyLoss loss_;
};

#endif // PROJECTNAME_SOFTMAX_CUH
