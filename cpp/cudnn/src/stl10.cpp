//
// Created by Maksim Levental on 11/10/20.
//

#include <stl10.h>

#include "npy.h"
#include <cassert>
#include <filesystem>
#include <utility>

void STL10::load_data() {
    if (!(std::filesystem::exists(dataset_fp_) && std::filesystem::exists(label_fp_))) {
        std::cout << "Download dataset first!!" << std::endl;
        exit(EXIT_FAILURE);
    }
    vector<unsigned long> shape;
    bool fortran_order;
    vector<uint8_t> data;
    npy::LoadArrayFromNumpy(dataset_fp_, shape, fortran_order, data);
    // n x c x h x w
    assert(shape.size() == 4);
    int n_samples = shape[0];
    channels_ = shape[1], height_ = shape[2], width_ = shape[3];
    assert(channels_ == 3);
    assert(height_ == 96);
    assert(width_ == 96);

    auto num_pixels = channels_ * height_ * width_;
    for (int i = 0; i < n_samples; i++) {
        std::vector<float> image(&data[i * num_pixels], &data[(i + 1) * num_pixels]);
        data_pool_.push_back(image);
    }

    num_batches_ = (int)(n_samples / batch_size_);
    std::cout << "num_batches: " << num_batches_ << std::endl;
    std::cout << "loaded " << data_pool_.size() << " items.." << std::endl;
}

void STL10::load_target() {
    vector<unsigned long> shape;
    bool fortran_order;
    vector<uint8_t> data;

    npy::LoadArrayFromNumpy(label_fp_, shape, fortran_order, data);
    assert(shape.size() == 1);
    int n_targets = shape[0];
    assert(n_targets == data_pool_.size());

    // prepare input buffer for label
    // read all labels and converts to one-hot encoding
    for (int i = 0; i < n_targets; i++) {
        std::vector<float> target_batch(num_classes_, 0.f);
        target_batch[static_cast<int>(data[i])] = 1.f;
        target_pool_.push_back(target_batch);
    }
}

void STL10::normalize_data() {
<<<<<<< HEAD
    for (auto &sample : data_pool_) {
        float *sample_data_ptr = sample.data();
        for (int j = 0; j < channels_ * height_ * width_; j++) {
            sample_data_ptr[j] /= 255.f;
=======
    for (auto image : data_pool_) {
        float *image_ptr = image.data();
        for (int j = 0; j < channels_ * height_ * width_; j++) {
            image_ptr[j] /= 255.f;
>>>>>>> f5fcd8a... b&w images have only one channel 🤦
        }
    }
}

STL10::STL10(
    const string &dataset_fp, const string &label_fp, bool shuffle, int batch_size, int num_classes)
    : Dataset(dataset_fp, label_fp, shuffle, batch_size, num_classes) {

    STL10::load_data();
    STL10::normalize_data();
    STL10::load_target();

    if (shuffle_)
        shuffle_dataset();
    create_shared_space();
}

int STL10::get_num_batches() const { return num_batches_; }
