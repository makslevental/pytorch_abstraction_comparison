//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_ACTIVATION_CUH
#define PROJECTNAME_ACTIVATION_CUH

#include <layer.h>

class Activation : public Layer {
public:
    Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f);
    ~Activation() override;

    Tensor<float> *forward(Tensor<float> *input) override;
    Tensor<float> *backward(Tensor<float> *grad_input) override;

private:
    void fwd_initialize(Tensor<float> *input) override;

    cudnnActivationDescriptor_t act_desc_;
    cudnnActivationMode_t act_mode_;
    float act_coef_;
};

#endif // PROJECTNAME_ACTIVATION_CUH
