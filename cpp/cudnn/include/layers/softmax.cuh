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

    Tensor<double> *forward(Tensor<double> *input) override;
    Tensor<double> *backward(Tensor<double> *grad_of_output) override;
};

#endif // PROJECTNAME_SOFTMAX_CUH
