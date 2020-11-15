#ifndef _MNIST_H_
#define _MNIST_H_

#include "dataset.h"
#include "tensor.h"

#define NUMBER_MNIST_CLASSES 10

template <typename dtype> class MNIST : public Dataset<dtype> {
public:
    MNIST(
        const string &dataset_fp,
        const string &label_fp,
        bool shuffle,
        int batch_size,
        int num_classes);
    [[nodiscard]] int get_num_batches() const override;

protected:
    void normalize_data() override;
    void load_data() override;
    void load_target() override;
};

template <typename dtype> class MNIST_DEBUG : public MNIST<dtype> {
public:
    MNIST_DEBUG(
        const string &datasetFp, const string &labelFp, bool shuffle, int batchSize, int numClasses)
        : MNIST<dtype>(datasetFp, labelFp, shuffle, batchSize, numClasses) {}
    std::tuple<Tensor<dtype> *, Tensor<dtype> *> get_next_batch() override {
        Tensor<dtype> *train_data, *train_target;
        std::tie(train_data, train_target) = Dataset<dtype>::get_next_batch();

        std::vector<double> target_one_hot(this->num_classes_, 0.f);
        target_one_hot[0] = 1.f;
        for (int batch = 0; batch < this->batch_size_; batch++) {
            std::copy(
                target_one_hot.begin(),
                target_one_hot.end(),
                &this->target_->get_host_ptr()[this->num_classes_ * batch]);
        }
        return std::make_tuple(train_data, this->target_);
    }
};

#endif // _MNIST_H_
