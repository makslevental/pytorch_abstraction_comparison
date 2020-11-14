#ifndef _MNIST_H_
#define _MNIST_H_

#include "dataset.h"
#include "tensor.h"

#define NUMBER_MNIST_CLASSES 10

class MNIST : public Dataset {
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

class MNIST_DEBUG : public MNIST {
public:
    MNIST_DEBUG(
        const string &datasetFp, const string &labelFp, bool shuffle, int batchSize, int numClasses)
        : MNIST(datasetFp, labelFp, shuffle, batchSize, numClasses) {}
    std::tuple<Tensor<double> *, Tensor<double> *> get_next_batch() override {
        Tensor<double> *train_data, *train_target;
        std::tie(train_data, train_target) = Dataset::get_next_batch();

        std::vector<double> target_one_hot(num_classes_, 0.f);
        target_one_hot[0] = 1.f;
        for (int batch = 0; batch < batch_size_; batch++) {
            std::copy(
                target_one_hot.begin(),
                target_one_hot.end(),
                &target_->get_host_ptr()[num_classes_ * batch]);
        }
        return std::make_tuple(train_data, target_);
    }
};

#endif // _MNIST_H_
