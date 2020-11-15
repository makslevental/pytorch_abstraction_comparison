//
// Created by Maksim Levental on 11/10/20.
//

#ifndef PROJECTNAME_DATASET_H
#define PROJECTNAME_DATASET_H

#include "tensor.h"
#include <vector>

using namespace std;

template <typename dtype>
class Dataset {
public:
    Dataset(
        string dataset_fp,
        string label_fp,
        bool shuffle,
        int batch_size,
        int num_classes);

    virtual std::tuple<Tensor<dtype> *, Tensor<dtype> *> get_next_batch();
    [[nodiscard]] virtual int get_num_batches() const = 0;
    virtual int len();
    virtual void reset();
    void test_dataset();

protected:
    std::string dataset_fp_;
    std::string label_fp_;

    std::vector<std::vector<dtype>> data_pool_;
    std::vector<std::vector<dtype>> target_pool_;
    Tensor<dtype> *data_ = nullptr;
    Tensor<dtype> *target_ = nullptr;

    virtual void load_data() = 0;
    virtual void load_target() = 0;
    virtual void normalize_data() = 0;
    void create_shared_space();
    void shuffle_dataset();

    int current_batch_;
    bool shuffle_;
    int batch_size_;
    int channels_;
    int height_;
    int width_;
    int num_classes_;
    int num_batches_;
};

#endif // PROJECTNAME_DATASET_H
