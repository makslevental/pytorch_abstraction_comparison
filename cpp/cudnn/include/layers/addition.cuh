//
// Created by Maksim Levental on 11/5/20.
//

#ifndef PROJECTNAME_ADDITION_CUH
#define PROJECTNAME_ADDITION_CUH

#include "layer.h"

class Addition : public Layer {
public:
    ~Addition() override;
    Tensor<float> *add(Tensor<float> *A, Tensor<float> *B);
    Tensor<float> *forward(Tensor<float> *input) override;
    Tensor<float> *backward(Tensor<float> *grad_input) override;

protected:
    void fwd_initialize(Tensor<float> *input) override;

private:
    cudnnOpTensorDescriptor_t op_descriptor = nullptr;
};

#endif // PROJECTNAME_ADDITION_CUH
