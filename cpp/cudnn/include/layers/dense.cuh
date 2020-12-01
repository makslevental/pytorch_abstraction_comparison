//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_DENSE_CUH
#define PROJECTNAME_DENSE_CUH

#include "layer.h"

template <typename dtype> class Dense : public Layer<dtype> {
public:
    Dense(std::string name, int out_size);
    ~Dense() override;

    Tensor<dtype> *forward(Tensor<dtype> *input) override;
    Tensor<dtype> *backward(Tensor<dtype> *grad_input) override;

private:
    void fwd_initialize(Tensor<dtype> *input) override;
    void bwd_initialize(Tensor<dtype> *grad_output) override;
    std::tuple<int, int> calculate_fan_in_and_fan_out() override;

    int output_size_ = 0;

    dtype *d_one_vec = nullptr;
    int input_num_features_;
};

#endif // PROJECTNAME_DENSE_CUH
