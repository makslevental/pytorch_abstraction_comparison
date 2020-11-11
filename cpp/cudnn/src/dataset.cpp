//
// Created by Maksim Levental on 11/10/20.
//

#include "dataset.h"
#include <cassert>
#include <random>
#include <utility>

using namespace std;

void Dataset::create_shared_space() {
    // create Tensors with batch size and sample size
    data_ = new Tensor<float>(batch_size_, channels_, height_, width_);
    target_ = new Tensor<float>(batch_size_, num_classes_);
}

void Dataset::shuffle_dataset() {
    std::random_device rd;
    std::mt19937 g_data(rd());
    auto g_target = g_data;

    std::shuffle(std::begin(data_pool_), std::end(data_pool_), g_data);
    std::shuffle(std::begin(target_pool_), std::end(target_pool_), g_target);
}
Dataset::Dataset(string dataset_fp, string label_fp, bool shuffle, int batch_size, int num_classes)
    : dataset_fp_(std::move(dataset_fp)), label_fp_(std::move(label_fp)), shuffle_(shuffle),
      batch_size_(batch_size), num_classes_(num_classes) {

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
    assert(num_batches_ > -1);
    int data_idx = (current_batch_ * batch_size_) % num_batches_;
    int data_size = channels_ * width_ * height_;
    assert(data_size == data_->size());
    assert(num_classes_ == target_->size());

    for (int batch = 0, sample = data_idx; batch < batch_size_;
         batch++, sample = data_idx + batch) {
        std::copy(
            data_pool_[sample].begin(),
            data_pool_[sample].end(),
            &data_->get_host_ptr()[data_size * batch]);
        std::copy(
            target_pool_[sample].begin(),
            target_pool_[sample].end(),
            &target_->get_host_ptr()[num_classes_ * batch]);
    }

    current_batch_++;
    return std::make_tuple(data_, target_);
}
void Dataset::test_dataset() {
    Tensor<float> *train_data, *train_target;
    std::tie(train_data, train_target) = get_next_batch();
    train_data->print("train_data", true);
    train_target->print("train_data", true);
    reset();
}
