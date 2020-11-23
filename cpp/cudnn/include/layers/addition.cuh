//
// Created by Maksim Levental on 11/5/20.
//

#ifndef PROJECTNAME_ADDITION_CUH
#define PROJECTNAME_ADDITION_CUH

#include "layer.h"

template <typename dtype> class Addition : public Layer<dtype> {
public:
    ~Addition() override;
    explicit Addition(std::string name);
    Tensor<dtype> *add(Tensor<dtype> *A, Tensor<dtype> *B);
    Tensor<dtype> *forward(Tensor<dtype> *input) override;
    Tensor<dtype> *backward(Tensor<dtype> *grad_input) override;
    int get_num_params() override { return 0; }

protected:
    void fwd_initialize(Tensor<dtype> *input) override;

private:
    cudnnOpTensorDescriptor_t op_descriptor = nullptr;
};

#endif // PROJECTNAME_ADDITION_CUH
