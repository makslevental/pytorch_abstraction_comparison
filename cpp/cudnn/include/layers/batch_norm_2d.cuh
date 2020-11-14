//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_BATCH_NORM_2D_CUH
#define PROJECTNAME_BATCH_NORM_2D_CUH

#include "layer.h"

class BatchNorm2d : public Layer {
public:
    explicit BatchNorm2d(
        std::string name,
        double epsilon = 1e-5,
        double momentum = 0.1,
        bool affine = true,
        bool track_running_stats = true,
        cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    ~BatchNorm2d() override;

    Tensor<double> *forward(Tensor<double> *input) override;
    Tensor<double> *backward(Tensor<double> *grad_output) override;

private:
    void fwd_initialize(Tensor<double> *input) override;
    void bwd_initialize(Tensor<double> *grad_output) override;

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
    Tensor<double> *running_mean_ = nullptr;
    Tensor<double> *running_var_ = nullptr;
    Tensor<double> *save_mean_ = nullptr;
    Tensor<double> *save_var_ = nullptr;

    cudnnBatchNormMode_t mode_;
    cudnnTensorDescriptor_t derived_bn_scale_bias_mean_var_desc_;
};

#endif // PROJECTNAME_BATCH_NORM_2D_CUH

//TODO: fused ops