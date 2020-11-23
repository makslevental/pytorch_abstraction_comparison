//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_SOFTMAX_CUH
#define PROJECTNAME_SOFTMAX_CUH

#include "layer.h"

template <typename dtype> class Softmax : public Layer<dtype> {
public:
    explicit Softmax(std::string name);
    ~Softmax() override = default;

    Tensor<dtype> *forward(Tensor<dtype> *input) override;
    Tensor<dtype> *backward(Tensor<dtype> *grad_input) override;
    int get_num_params() override { return 0; }

protected:
    void fwd_initialize(Tensor<dtype> *input) override;
    void bwd_initialize(Tensor<dtype> *grad_output) override;
};

#endif // PROJECTNAME_SOFTMAX_CUH
