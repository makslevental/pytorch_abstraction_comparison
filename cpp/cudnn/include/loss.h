#ifndef _LOSS_H_
#define _LOSS_H_

#include "tensor.h"

template <typename dtype>
class CrossEntropyLoss {
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    dtype loss(Tensor<dtype> *predict, Tensor<dtype> *target);

private:
    // reduced loss
    dtype h_loss_ = 0.f;
    dtype *d_loss_ = nullptr;

    dtype *d_workspace_ = nullptr;
    void init_workspace(int batch_size);
};

#endif // _LOSS_H_