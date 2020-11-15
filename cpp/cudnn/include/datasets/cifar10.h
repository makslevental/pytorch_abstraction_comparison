//
// Created by Maksim Levental on 11/13/20.
//

#ifndef PROJECTNAME_CIFAR10_H
#define PROJECTNAME_CIFAR10_H

#include "dataset.h"

#define NUMBER_CIFAR10_CLASSES 10

template <typename dtype> class CIFAR10 : public Dataset<dtype> {
public:
    CIFAR10(
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

template <typename dtype> class CIFAR10_DEBUG : public CIFAR10<dtype> {
public:
    CIFAR10_DEBUG(
        const string &datasetFp, const string &labelFp, bool shuffle, int batchSize, int numClasses)
        : CIFAR10<dtype>(datasetFp, labelFp, shuffle, batchSize, numClasses) {}
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

#endif // PROJECTNAME_CIFAR10_H
