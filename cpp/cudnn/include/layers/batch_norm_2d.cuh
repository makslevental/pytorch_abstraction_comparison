//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_BATCH_NORM_2D_CUH
#define PROJECTNAME_BATCH_NORM_2D_CUH

#include "layer.h"

template <typename dtype> class BatchNorm2d : public Layer<dtype> {
public:
    explicit BatchNorm2d(
        std::string name,
        double epsilon = 1e-5,
        double momentum = 1.0,
        bool affine = true,
        bool track_running_stats = true,
        cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT);

    Tensor<dtype> *forward(Tensor<dtype> *input) override;
    Tensor<dtype> *backward(Tensor<dtype> *grad_output) override;

private:
    void fwd_initialize(Tensor<dtype> *input) override;
    void bwd_initialize(Tensor<dtype> *grad_output) override;

    size_t workspace_size_ = 0;
    size_t reserve_size_ = 0;
    void **device_workspace_ = nullptr;
    void **device_reserve_space_ = nullptr;
    virtual void set_workspace();

    int num_features_;
    double epsilon_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;
    Tensor<dtype> *running_mean_ = nullptr;
    Tensor<dtype> *running_var_ = nullptr;
    Tensor<dtype> *save_mean_ = nullptr;
    Tensor<dtype> *save_var_ = nullptr;

    cudnnBatchNormMode_t mode_;
    cudnnTensorDescriptor_t derived_bn_scale_bias_mean_var_desc_;
};

#endif // PROJECTNAME_BATCH_NORM_2D_CUH

// TODO: fused ops