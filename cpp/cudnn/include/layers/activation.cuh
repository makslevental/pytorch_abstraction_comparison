//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_ACTIVATION_CUH
#define PROJECTNAME_ACTIVATION_CUH

#include "layer.h"

class Activation : public Layer {
public:
    Activation(std::string name, cudnnActivationMode_t mode, double coef = 0.f);
    ~Activation() override;

    Tensor<double> *forward(Tensor<double> *input) override;
    Tensor<double> *backward(Tensor<double> *grad_input) override;

private:
    cudnnActivationDescriptor_t act_desc_;
    cudnnActivationMode_t act_mode_;
    double act_coef_;
};

#endif // PROJECTNAME_ACTIVATION_CUH
