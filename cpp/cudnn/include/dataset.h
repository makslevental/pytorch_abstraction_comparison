//
// Created by Maksim Levental on 11/10/20.
//

#ifndef PROJECTNAME_DATASET_H
#define PROJECTNAME_DATASET_H

#include "tensor.h"
#include <vector>

using namespace std;

class Dataset {
public:
    Dataset(
        string dataset_fp,
        string label_fp,
        bool shuffle,
        int batch_size,
        int channels,
        int height,
        int width,
        int num_classes);

    std::tuple<Tensor<float> *, Tensor<float> *> get_next_batch();
    [[nodiscard]] virtual int get_num_batches() const = 0;
    virtual int len();
    virtual void reset();

protected:
    std::string dataset_fp_;
    std::string label_fp_;

    std::vector<std::vector<float>> data_pool_;
    std::vector<std::vector<float>> target_pool_;
    Tensor<float> *data_ = nullptr;
    Tensor<float> *target_ = nullptr;

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
