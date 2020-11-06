#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <string>
#include <vector>

#include <cudnn.h>

#include "helper.h"
#include "layer.h"
#include "loss.h"

typedef enum { training, inference } WorkloadType;

class Network {
public:
    Network();
    ~Network();

    void add_layer(Layer *layer);

    virtual Tensor<float> *forward(Tensor<float> *input);
    virtual void backward(Tensor<float> *input);
    void update(float learning_rate = 0.02f);

    int load_pretrain();
    int write_file();

    void cuda();
    void train();
    void eval();

    Tensor<float> *output_;

protected:
    std::vector<Layer *> layers_;
    CudaContext *cuda_ = nullptr;
    WorkloadType phase_ = inference;
};

#endif // _NETWORK_H_