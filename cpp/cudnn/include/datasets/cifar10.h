//
// Created by Maksim Levental on 11/13/20.
//

#ifndef PROJECTNAME_CIFAR10_H
#define PROJECTNAME_CIFAR10_H

#include "dataset.h"

#define NUMBER_CIFAR10_CLASSES 10

class CIFAR10 : public Dataset {
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

class CIFAR10_DEBUG : public CIFAR10 {
public:
    CIFAR10_DEBUG(
        const string &datasetFp, const string &labelFp, bool shuffle, int batchSize, int numClasses)
        : CIFAR10(datasetFp, labelFp, shuffle, batchSize, numClasses) {}
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

#endif // PROJECTNAME_CIFAR10_H
