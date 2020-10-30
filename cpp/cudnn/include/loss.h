#ifndef _LOSS_H_
#define _LOSS_H_

#include "tensor.h"

class CrossEntropyLoss {
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    float loss(Tensor<float> *predict, Tensor<float> *target);

private:
    // reduced loss
    float h_loss_ = 0.f;
    float *d_loss_ = nullptr;

    float *d_workspace_ = nullptr;
    void init_workspace(int batch_size);
};

#endif // _LOSS_H_