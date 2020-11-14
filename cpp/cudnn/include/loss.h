#ifndef _LOSS_H_
#define _LOSS_H_

#include "tensor.h"

class CrossEntropyLoss {
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    double loss(Tensor<double> *predict, Tensor<double> *target);

private:
    // reduced loss
    double h_loss_ = 0.f;
    double *d_loss_ = nullptr;

    double *d_workspace_ = nullptr;
    void init_workspace(int batch_size);
};

#endif // _LOSS_H_