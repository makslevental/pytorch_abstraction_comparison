//
// Created by Maksim Levental on 11/10/20.
//

#include <datasets/stl10.h>

#include "npy.h"
#include <cassert>
#include <utility>

template <typename dtype> void STL10<dtype>::load_data() {
    vector<unsigned long> shape;
    bool fortran_order;
    vector<uint8_t> data_buffer;
    npy::LoadArrayFromNumpy(this->dataset_fp_, shape, fortran_order, data_buffer);
    // n x c x h x w
    assert(shape.size() == 4);
    int n_samples = shape[0];
    this->channels_ = shape[1], this->height_ = shape[2], this->width_ = shape[3];
    assert(this->channels_ == 3);
    assert(this->height_ == 96);
    assert(this->width_ == 96);

    auto num_pixels = this->channels_ * this->height_ * this->width_;
    for (int i = 0; i < n_samples; i++) {
        uint32_t start_index = i * num_pixels;
        uint32_t image_start = start_index + 1;
        uint32_t image_end = image_start + num_pixels;
        std::vector<dtype> image(
            data_buffer.begin() + image_start, data_buffer.begin() + image_end);
        this->data_pool_.push_back(image);
    }

    this->num_batches_ = (int)(n_samples / this->batch_size_);
    std::cout << "num_batches: " << this->num_batches_ << std::endl;
    std::cout << "loaded " << this->data_pool_.size() << " items.." << std::endl;
}

template <typename dtype> void STL10<dtype>::load_target() {
    vector<unsigned long> label_shape;
    bool fortran_order;
    vector<uint8_t> label_buffer;

    npy::LoadArrayFromNumpy(this->label_fp_, label_shape, fortran_order, label_buffer);
    assert(label_shape.size() == 1);
    int n_targets = label_shape[0];
    assert(n_targets == this->data_pool_.size());

    // read all labels and converts to one-hot encoding
    for (int i = 0; i < n_targets; i++) {
        std::vector<dtype> target(this->num_classes_, 0.f);
        target[static_cast<int>(label_buffer[i])] = 1.f;
        this->target_pool_.push_back(target);
    }
}

template <typename dtype> void STL10<dtype>::normalize_data() {
    for (auto &sample : this->data_pool_) {
        std::transform(
            sample.begin(), sample.end(), sample.begin(), [](float i) { return i / 255.f; });
    }
}

template <typename dtype>
STL10<dtype>::STL10(
    const string &dataset_fp, const string &label_fp, bool shuffle, int batch_size, int num_classes)
    : Dataset<dtype>(dataset_fp, label_fp, shuffle, batch_size, num_classes) {

    STL10::load_data();
    STL10::normalize_data();
    STL10::load_target();

    if (this->shuffle_)
        this->shuffle_dataset();
    this->create_shared_space();
}

template <typename dtype> int STL10<dtype>::get_num_batches() const { return this->num_batches_; }

template class STL10<float>;
template class STL10<double>;
