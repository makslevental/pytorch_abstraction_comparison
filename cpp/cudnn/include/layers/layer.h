#ifndef _LAYER_H_
#define _LAYER_H_

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "cross_entropy_loss.h"
#include "helper.h"
#include "tensor.h"

class Layer {
public:
    Layer();
    virtual ~Layer();

    virtual Tensor<double> *forward(Tensor<double> *input) = 0;
    virtual Tensor<double> *backward(Tensor<double> *grad_of_input) = 0;

    std::string get_name() { return name_; }

    void set_cuda_context(CudaContext *context) { cuda_ = context; }

    void set_load_pretrain() { load_pretrain_ = true; };
    void set_gradient_stop() { gradient_stop_ = true; }

    /* Weight Freeze or Unfreeze */
    void freeze() { freeze_ = true; }
    void unfreeze() { freeze_ = false; }
    void train() { train_ = true; }
    void eval() { train_ = false; }

protected:
    virtual void fwd_initialize(Tensor<double> *input);
    virtual void bwd_initialize(Tensor<double> *grad_of_output);
    virtual void zero_out();

    // name of layer
    std::string name_;

    // tensor descriptor for the input/output tensors
    cudnnTensorDescriptor_t input_desc_ = nullptr;
    cudnnTensorDescriptor_t output_desc_ = nullptr;
    // weight/bias descriptor
    cudnnFilterDescriptor_t filter_desc_ = nullptr;
    cudnnTensorDescriptor_t bias_desc_ = nullptr;

    // output memory
    Tensor<double> *input_ = nullptr;       /* x  */
    Tensor<double> *output_ = nullptr;      /* y  */
    Tensor<double> *grad_of_input_ = nullptr;  /* dx */
    Tensor<double> *grad_of_output_ = nullptr; /* dy */
    int input_size_;

    // master weights & bias
    bool freeze_ = false; /* control parameter updates */
    bool train_ = false;
    Tensor<double> *weights_ = nullptr;      /* w */
    Tensor<double> *biases_ = nullptr;       /* b */
    Tensor<double> *grad_weights_ = nullptr; /* dw */
    Tensor<double> *grad_biases_ = nullptr;  /* db */

    int batch_size_ = 0; // mini-batch size

    // initialize weights along with the input size
    void init_weight_bias(unsigned int seed = 0);
    void update_weights_biases(double learning_rate);

    // get_device_ptr handle container
    CudaContext *cuda_ = nullptr;

    // pretrain parameters
    bool load_pretrain_ = false;
    int load_parameter();
    int save_parameter();

    // gradient stop tagging
    bool gradient_stop_ = false;

    friend class Network;
};

#endif // _LAYER_H_