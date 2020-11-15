//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_ACTIVATION_CUH
#define PROJECTNAME_ACTIVATION_CUH

#include "layer.h"

template <typename dtype>
class Activation : public Layer<dtype> {
public:
    Activation(std::string name, cudnnActivationMode_t mode, double coef = 0.f);
    ~Activation() override;

    Tensor<dtype> *forward(Tensor<dtype> *input) override;
    Tensor<dtype> *backward(Tensor<dtype> *grad_input) override;

private:
    cudnnActivationDescriptor_t act_desc_;
    cudnnActivationMode_t act_mode_;
    double act_coef_;
};

#endif // PROJECTNAME_ACTIVATION_CUH
