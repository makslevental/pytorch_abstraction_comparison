//
// Created by Maksim Levental on 10/29/20.
//

#ifndef PROJECTNAME_CONV_2D_CUH
#define PROJECTNAME_CONV_2D_CUH

#include "layer.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>

template <typename dtype> class Conv2d : public Layer<dtype> {
public:
    Conv2d(
        std::string name,
        int out_channels,
        int kernel_size,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        bool bias = true);
    ~Conv2d() override;

    Tensor<dtype> *forward(Tensor<dtype> *input) override;
    Tensor<dtype> *backward(Tensor<dtype> *grad_output) override;

private:
    void fwd_initialize(Tensor<dtype> *input) override;
    void bwd_initialize(Tensor<dtype> *grad_output) override;
    std::tuple<int, int> calculate_fan_in_and_fan_out() override;

    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;
    bool bias_;

    std::array<int, 4> output_size_;
    // convolution
    cudnnConvolutionDescriptor_t conv_desc_;

    cudnnConvolutionFwdAlgo_t conv_fwd_algo_;
    cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo_;
    cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo_;

    size_t workspace_size_ = 0;
    void **device_workspace_ = nullptr;
    virtual void set_workspace();
};

#endif // PROJECTNAME_CONV_2D_CUH
