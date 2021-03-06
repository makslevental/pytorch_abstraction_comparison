#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <string>
#include <vector>

#include <cudnn.h>

#include <cuda_helper.h>
#include "layers/layer.h"
#include "loss.h"

typedef enum { training, inference } WorkloadType;

template <typename dtype> class Network {
public:
    Network();
    ~Network();

    void add_layer(Layer<dtype> *layer);

    virtual Tensor<dtype> *forward(Tensor<dtype> *input);
    virtual void backward(Tensor<dtype> *input);
    void update(double learning_rate = 0.02f);

    int load_pretrain();
    int write_file();

    void cuda();
    void train();
    void eval();
    void zero_grad();
    void print_all_params();

    Tensor<dtype> *output_;

    void print_all_grads();
protected:
    std::vector<Layer<dtype> *> layers_;
    CudaContext<dtype> *cuda_ = nullptr;
    WorkloadType phase_ = inference;
};

#endif // _NETWORK_H_