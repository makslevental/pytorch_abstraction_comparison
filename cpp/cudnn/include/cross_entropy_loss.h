#ifndef _CROSS_ENTROPY_LOSS_H_
#define _CROSS_ENTROPY_LOSS_H_

#include "tensor.h"

class CrossEntropyLoss {
public:
    explicit CrossEntropyLoss(int batch_size, CudaContext *cuda_context);
    ~CrossEntropyLoss();

    double loss(Tensor<double> *predict, Tensor<double> *target);
    Tensor<double> *backward();

private:
    // reduced loss
    double *d_loss_ = nullptr;
    double *d_workspace_ = nullptr;
    Tensor<double> *predict_ = nullptr;
    Tensor<double> *grad_of_predict_ = nullptr;
    Tensor<double> *target_ = nullptr;
    cudnnOpTensorDescriptor_t op_descriptor_ = nullptr;
    CudaContext *cuda_ = nullptr;
    double scale_;
};

#endif // _LOSS_H_