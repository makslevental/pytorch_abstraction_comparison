#ifndef _LAYER_H_
#define _LAYER_H_

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "helper.h"
#include "loss.h"
#include "tensor.h"

template <typename dtype> class Layer {
public:
    Layer();
    virtual ~Layer();

    virtual Tensor<dtype> *forward(Tensor<dtype> *input) = 0;
    virtual Tensor<dtype> *backward(Tensor<dtype> *grad_input) = 0;

    std::string get_name() { return name_; }

    void set_cuda_context(CudaContext<dtype> *context) { cuda_ = context; }

    void set_load_pretrain() { load_pretrain_ = true; };
    void set_gradient_stop() { gradient_stop_ = true; }

    /* Weight Freeze or Unfreeze */
    void freeze() { freeze_ = true; }
    void unfreeze() { freeze_ = false; }
    void train() { train_ = true; }
    void eval() { train_ = false; }
    void update_weights_biases(dtype learning_rate);
    void zero_out();

protected:
    virtual void fwd_initialize(Tensor<dtype> *input);
    virtual void bwd_initialize(Tensor<dtype> *grad_output);

    // name of layer
    std::string name_;

    // tensor descriptor for the input/output tensors
    cudnnTensorDescriptor_t input_desc_ = nullptr;
    cudnnTensorDescriptor_t output_desc_ = nullptr;
    // weight/bias descriptor
    cudnnFilterDescriptor_t filter_desc_ = nullptr;
    cudnnTensorDescriptor_t bias_desc_ = nullptr;

    // output memory
    Tensor<dtype> *input_ = nullptr;          /* x  */
    Tensor<dtype> *output_ = nullptr;         /* y  */
    Tensor<dtype> *grad_of_input_ = nullptr;  /* dx */
    Tensor<dtype> *grad_of_output_ = nullptr; /* dy */
    int input_size_;

    // master weights & bias
    bool freeze_ = false; /* control parameter updates */
    bool train_ = false;
    Tensor<dtype> *weights_ = nullptr;      /* w */
    Tensor<dtype> *biases_ = nullptr;       /* b */
    Tensor<dtype> *grad_weights_ = nullptr; /* dw */
    Tensor<dtype> *grad_biases_ = nullptr;  /* db */

    int batch_size_ = 0; // mini-batch size

    // initialize weights along with the input size
    void init_weight_bias(unsigned int seed = 0);

    // get_device_ptr handle container
    CudaContext<dtype> *cuda_ = nullptr;

    // pretrain parameters
    bool load_pretrain_ = false;
    int load_parameter();
    int save_parameter();

    // gradient stop_ tagging
    bool gradient_stop_ = false;

    template <typename ndtype> friend class Network;
};

#endif // _LAYER_H_