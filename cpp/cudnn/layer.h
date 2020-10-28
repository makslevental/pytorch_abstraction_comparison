#ifndef _LAYER_H_
#define _LAYER_H_

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "helper.h"
#include "loss.h"
#include "tensor.h"

class Layer {
public:
    Layer();
    virtual ~Layer();

    virtual Tensor<float> *forward(Tensor<float> *input) = 0;
    virtual Tensor<float> *backward(Tensor<float> *grad_input) = 0;

    std::string get_name() { return name_; }

    virtual float get_loss(Tensor<float> *target);
    virtual int get_accuracy(Tensor<float> *target);

    void set_cuda_context(CudaContext *context) { cuda_ = context; }

    void set_load_pretrain() { load_pretrain_ = true; };
    void set_gradient_stop() { gradient_stop_ = true; }

    /* Weight Freeze or Unfreeze */
    void freeze() { freeze_ = true; }
    void unfreeze() { freeze_ = false; }

protected:
    virtual void fwd_initialize(Tensor<float> *input) = 0;
    virtual void bwd_initialize(Tensor<float> *grad_output) = 0;

    // name of layer
    std::string name_;

    // tensor descriptor for the input/output tensors
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    // weight/bias descriptor
    cudnnFilterDescriptor_t filter_desc_;
    cudnnTensorDescriptor_t bias_desc_;

    // output memory
    Tensor<float> *input_ = nullptr;       /* x  */
    Tensor<float> *output_ = nullptr;      /* y  */
    Tensor<float> *grad_input_ = nullptr;  /* dx */
    Tensor<float> *grad_output_ = nullptr; /* dy */

    // master weights & bias
    bool freeze_ = false;                   /* control parameter updates */
    Tensor<float> *weights_ = nullptr;      /* w */
    Tensor<float> *biases_ = nullptr;       /* b */
    Tensor<float> *grad_weights_ = nullptr; /* dw */
    Tensor<float> *grad_biases_ = nullptr;  /* db */

    int batch_size_ = 0; // mini-batch size

    // initialize weights along with the input size
    void init_weight_bias(unsigned int seed = 0);
    void update_weights_biases(float learning_rate);

    // cuda handle container
    CudaContext *cuda_ = nullptr;

    // pretrain parameters
    bool load_pretrain_ = false;
    int load_parameter();
    int save_parameter();

    // gradient stop tagging
    bool gradient_stop_ = false;

    friend class Network;
};

class Dense : public Layer {
public:
    Dense(std::string name, int out_size);
    ~Dense() override;

    Tensor<float> *forward(Tensor<float> *input) override;
    Tensor<float> *backward(Tensor<float> *grad_input) override;

private:
    void fwd_initialize(Tensor<float> *input) override;
    void bwd_initialize(Tensor<float> *grad_output) override;

    int input_size_ = 0;
    int output_size_ = 0;

    float *d_one_vec = nullptr;
};

class Activation : public Layer {
public:
    Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f);
    ~Activation() override;

    Tensor<float> *forward(Tensor<float> *input) override;
    Tensor<float> *backward(Tensor<float> *grad_input) override;

private:
    void fwd_initialize(Tensor<float> *input) override;
    void bwd_initialize(Tensor<float> *grad_output) override;

    cudnnActivationDescriptor_t act_desc_;
    cudnnActivationMode_t act_mode_;
    float act_coef_;
};

class Softmax : public Layer {
public:
    explicit Softmax(std::string name);
    ~Softmax() override;

    Tensor<float> *forward(Tensor<float> *input) override;
    Tensor<float> *backward(Tensor<float> *grad_input) override;

    float get_loss(Tensor<float> *target) override;
    int get_accuracy(Tensor<float> *target) override;

protected:
    void fwd_initialize(Tensor<float> *input) override;
    void bwd_initialize(Tensor<float> *grad_output) override;

    CrossEntropyLoss loss_;
};

class Conv2D : public Layer {
public:
    Conv2D(
        std::string name,
        int out_channels,
        int kernel_size,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        bool bias = false);
    ~Conv2D() override;

    Tensor<float> *forward(Tensor<float> *input) override;
    Tensor<float> *backward(Tensor<float> *grad_output) override;

private:
    void fwd_initialize(Tensor<float> *input) override;
    void bwd_initialize(Tensor<float> *grad_output) override;

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
    void **d_workspace_ = nullptr;
    virtual void set_workspace();
};

class Pooling : public Layer {
public:
    Pooling(std::string name, int kernel_size, int padding, int stride, cudnnPoolingMode_t mode);
    ~Pooling() override;

    Tensor<float> *forward(Tensor<float> *input) override;
    Tensor<float> *backward(Tensor<float> *grad_output) override;

private:
    void fwd_initialize(Tensor<float> *input) override;
    void bwd_initialize(Tensor<float> *grad_output) override;

    int kernel_size_;
    int padding_;
    int stride_;
    cudnnPoolingMode_t mode_;

    std::array<int, 4> output_size_;
    cudnnPoolingDescriptor_t pool_desc_;
};

#endif // _LAYER_H_