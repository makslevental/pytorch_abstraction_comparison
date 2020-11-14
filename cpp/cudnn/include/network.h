#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <string>
#include <vector>

#include <cudnn.h>

#include "helper.h"
#include "layers/layer.h"
#include "loss.h"

typedef enum { training, inference } WorkloadType;

class Network {
public:
    Network();
    ~Network();

    void add_layer(Layer *layer);

    virtual Tensor<double> *forward(Tensor<double> *input);
    virtual void backward(Tensor<double> *input);
    void update(double learning_rate = 0.02f);

    int load_pretrain();
    int write_file();

    void cuda();
    void train();
    void eval();

    Tensor<double> *output_;

protected:
    std::vector<Layer *> layers_;
    CudaContext *cuda_ = nullptr;
    WorkloadType phase_ = inference;
};

#endif // _NETWORK_H_