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

    Tensor<float> *forward(Tensor<float> *input) override;
    Tensor<float> *backward(Tensor<float> *grad_input) override;

private:
    void fwd_initialize(Tensor<float> *input) override;
    void bwd_initialize(Tensor<float> *grad_output) override;

    int input_size_ = 0;
    int output_size_ = 0;

    float *d_one_vec = nullptr;
};

#endif // PROJECTNAME_DENSE_CUH
