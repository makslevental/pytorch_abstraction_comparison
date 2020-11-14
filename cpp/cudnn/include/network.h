#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <string>
#include <vector>

#include <cudnn.h>

#include "cross_entropy_loss.h"
#include "helper.h"
#include "layers/layer.h"

typedef enum { training, inference } WorkloadType;

class Network {
public:
    Network();
    ~Network();

    void add_layer(Layer *layer);

    virtual Tensor<double> *forward(Tensor<double> *input);
    virtual void backward(Tensor<double> *input);
    void update(double learning_rate = 0.02f);
    [[nodiscard]] CudaContext *get_cuda_context() const;

    int load_pretrain();
    int write_file();

    void cuda();
    void train();
    void eval();

    Tensor<double> *output_;

    Tensor<double> *forward(Tensor<double> *input, int DEBUG);

protected:
    std::vector<Layer *> layers_;
    CudaContext *cuda_ = nullptr;

protected:
    WorkloadType phase_ = inference;
};

#endif // _NETWORK_H_