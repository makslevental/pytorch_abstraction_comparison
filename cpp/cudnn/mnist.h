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

#define MNIST_CLASS 10

class MNIST {
public:
    MNIST()
        : dataset_dir_("/Users/maksim/dev_projects/pytorch_abstraction_comparison/data"),
          shuffle_(false) {}
    explicit MNIST(std::string dataset_dir)
        : dataset_dir_(std::move(dataset_dir)), shuffle_(false) {}
    ~MNIST();

    // load train dataset
    void train(int batch_size = 1, bool shuffle = false);

    // load test dataset
    void test(int batch_size = 1);

    // update shared batch data buffer at current step index
    void get_batch();
    // increase current step index
    // optionally it updates shared buffer if input parameter is true.
    int next();

    // returns a pointer which has input batch data
    Tensor<float> *get_data() { return data_; }
    // returns a pointer which has target batch data
    Tensor<float> *get_target() { return target_; }

private:
    // predefined file names
    std::string dataset_dir_;
    std::string train_dataset_file_ = "train-images-idx3-ubyte";
    std::string train_label_file_ = "train-labels-idx1-ubyte";
    std::string test_dataset_file_ = "t10k-images-idx3-ubyte";
    std::string test_label_file_ = "t10k-labels-idx1-ubyte";

    // container
    std::vector<std::vector<float>> data_pool_;
    std::vector<std::array<float, MNIST_CLASS>> target_pool_;
    Tensor<float> *data_ = nullptr;
    Tensor<float> *target_ = nullptr;

    // data loader initialization
    void load_data(std::string &image_file_path);
    void load_target(std::string &label_file_path);

    void normalize_data();

    static int to_int(const uint8_t *ptr);

    // data loader control
    int step_ = -1;
    bool shuffle_;
    int batch_size_ = 1;
    int channels_ = 1;
    int height_ = 1;
    int width_ = 1;
    int num_classes_ = 10;
    int num_steps_ = 0;

    void create_shared_space();
    void shuffle_dataset();
};

#endif // _MNIST_H_
