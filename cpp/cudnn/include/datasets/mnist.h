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

#endif // _MNIST_H_
