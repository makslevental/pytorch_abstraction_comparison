//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_POOLING_CUH
#define PROJECTNAME_POOLING_CUH

#include "layer.h"

class Pooling : public Layer {
public:
    Pooling(std::string name, int kernel_size, int padding, int stride, cudnnPoolingMode_t mode);
    ~Pooling() override;

    Tensor<float> *forward(Tensor<float> *input) override;
    Tensor<float> *backward(Tensor<float> *grad_output) override;

private:
    void fwd_initialize(Tensor<float> *input) override;

    int kernel_size_;
    int padding_;
    int stride_;
    cudnnPoolingMode_t mode_;

    std::array<int, 4> output_size_;
    cudnnPoolingDescriptor_t pool_desc_;
};

#endif // PROJECTNAME_POOLING_CUH
