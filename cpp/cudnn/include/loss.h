#ifndef _LOSS_H_
#define _LOSS_H_

#define DEBUG_LOSS 0

#include "tensor.h"

template <typename dtype> class CrossEntropyLoss {
public:
    explicit CrossEntropyLoss(int batch_size, int target_size);
    ~CrossEntropyLoss();

    dtype loss(Tensor<dtype> *predict, Tensor<dtype> *target);

private:
    dtype h_loss_ = 0.f;
    dtype *d_loss_ = nullptr;
    int num_sms;
    int num_blocks_per_sm;
    int num_blocks;

    dtype *d_workspace_ = nullptr;
    void init_workspace(int batch_size);
};

#endif // _LOSS_H_