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

#endif // PROJECTNAME_CIFAR10_H
