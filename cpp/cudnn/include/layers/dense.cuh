//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_DENSE_CUH
#define PROJECTNAME_DENSE_CUH

#include "layer.h"

class Dense : public Layer {
public:
    Dense(std::string name, int out_size);
    ~Dense() override;

    Tensor<double> *forward(Tensor<double> *input) override;
    Tensor<double> *backward(Tensor<double> *grad_input) override;

private:
    void fwd_initialize(Tensor<double> *input) override;
    void bwd_initialize(Tensor<double> *grad_output) override;

    int output_size_ = 0;

    double *d_one_vec = nullptr;
};

#endif // PROJECTNAME_DENSE_CUH
