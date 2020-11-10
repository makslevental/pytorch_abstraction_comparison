//
// Created by Maksim Levental on 11/10/20.
//

#include "dataset.h"
#include <random>
#include <utility>

using namespace std;

void Dataset::create_shared_space() {
    // create Tensors with batch size and sample size
    data_ = new Tensor<float>(batch_size_, channels_, height_, width_);
    data_->tensor_descriptor();
    target_ = new Tensor<float>(batch_size_, num_classes_);
}

void Dataset::shuffle_dataset() {
    std::random_device rd;
    std::mt19937 g_data(rd());
    auto g_target = g_data;

    std::shuffle(std::begin(data_pool_), std::end(data_pool_), g_data);
    std::shuffle(std::begin(target_pool_), std::end(target_pool_), g_target);
}
Dataset::Dataset(
    string dataset_fp,
    string label_fp,
    bool shuffle,
    int batch_size,
    int channels,
    int height,
    int width,
    int num_classes)
    : dataset_fp_(std::move(dataset_fp)), label_fp_(std::move(label_fp)), shuffle_(shuffle),
      batch_size_(batch_size), channels_(channels), height_(height), width_(width),
      num_classes_(num_classes) {

    current_batch_ = 0;
    num_batches_ = -1;
}

void Dataset::reset() {
    if (shuffle_)
        shuffle_dataset();
    current_batch_ = 0;
}

int Dataset::len() { return data_pool_.size(); }
std::tuple<Tensor<float> *, Tensor<float> *> Dataset::get_next_batch() {
    if (current_batch_ < 0) {
        std::cout << "You must initialize dataset first.." << std::endl;
        exit(EXIT_FAILURE);
    }
    //    std::cout << " internal step: " << current_batch_ << std::endl;

    // index cliping
    int data_idx = (current_batch_ * batch_size_) % num_batches_;

    int data_size = channels_ * width_ * height_;
    for (int i = 0; i < batch_size_; i++) {
        std::copy(
            data_pool_[data_idx + i].data(),
            &data_pool_[data_idx + i].data()[data_size],
            &data_->get_host_ptr()[data_size * i]);

        std::copy(
            target_pool_[data_idx + i].data(),
            &target_pool_[data_idx + i].data()[num_classes_],
            &target_->get_host_ptr()[num_classes_ * i]);
    }

    current_batch_++;
    return std::make_tuple(data_, target_);
}
