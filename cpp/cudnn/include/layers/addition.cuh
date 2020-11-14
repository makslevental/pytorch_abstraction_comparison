//
// Created by Maksim Levental on 11/5/20.
//

#ifndef PROJECTNAME_ADDITION_CUH
#define PROJECTNAME_ADDITION_CUH

#include "layer.h"

class Addition : public Layer {
public:
    ~Addition() override;
    Tensor<double> *add(Tensor<double> *A, Tensor<double> *B);
    Tensor<double> *forward(Tensor<double> *input) override;
    Tensor<double> *backward(Tensor<double> *grad_input) override;

protected:
    void fwd_initialize(Tensor<double> *input) override;

private:
    cudnnOpTensorDescriptor_t op_descriptor = nullptr;
};

#endif // PROJECTNAME_ADDITION_CUH
