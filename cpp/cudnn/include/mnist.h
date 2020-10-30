#ifndef _MNIST_H_
#define _MNIST_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "tensor.h"

#define NUMBER_MNIST_CLASSES 10

class MNIST {
public:
    MNIST()
        : dataset_dir_("/Users/maksim/dev_projects/pytorch_abstraction_comparison/data"),
          shuffle_(false) {}
    explicit MNIST(std::string dataset_dir)
        : dataset_dir_(std::move(dataset_dir)), shuffle_(false) {}
    ~MNIST();

    void train(int batch_size = 1, bool shuffle = false);

    void test(int batch_size = 1);

    std::tuple<Tensor<float> *, Tensor<float> *> get_next_batch();
    [[nodiscard]] int get_num_batches() const;
    int len();
    void reset();

private:
    // predefined file names
    std::string dataset_dir_;
    std::string train_dataset_file_ = "train-images-idx3-ubyte";
    std::string train_label_file_ = "train-labels-idx1-ubyte";
    std::string test_dataset_file_ = "t10k-images-idx3-ubyte";
    std::string test_label_file_ = "t10k-labels-idx1-ubyte";

    std::vector<std::vector<float>> data_pool_;
    std::vector<std::array<float, NUMBER_MNIST_CLASSES>> target_pool_;
    Tensor<float> *data_ = nullptr;
    Tensor<float> *target_ = nullptr;

    void load_data(std::string &image_file_path);
    void load_target(std::string &label_file_path);

    void normalize_data();

    static int to_int(const uint8_t *ptr);

    int current_batch_ = -1;
    bool shuffle_;
    int batch_size_ = 1;
    int channels_ = 1;
    int height_ = 1;
    int width_ = 1;
    // TODO: normalize this
    int num_classes_ = NUMBER_MNIST_CLASSES;
    int num_batches_ = 0;

private:
    void create_shared_space();
    void shuffle_dataset();
};

#endif // _MNIST_H_
