//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_POOLING_CUH
#define PROJECTNAME_POOLING_CUH

#include "layer.h"

template <typename dtype>
class Pooling : public Layer<dtype> {
public:
    Pooling(std::string name, int kernel_size, int padding, int stride, cudnnPoolingMode_t mode);
    ~Pooling() override;

    Tensor<dtype> *forward(Tensor<dtype> *input) override;
    Tensor<dtype> *backward(Tensor<dtype> *grad_output) override;

private:
    void fwd_initialize(Tensor<dtype> *input) override;

    int kernel_size_;
    int padding_;
    int stride_;
    cudnnPoolingMode_t mode_;
    std::array<int, 4> output_size_;
    cudnnPoolingDescriptor_t pool_desc_;
};

#endif // PROJECTNAME_POOLING_CUH

// TODO: avg pool