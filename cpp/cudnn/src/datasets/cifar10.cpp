//
// Created by Maksim Levental on 11/13/20.
//

#include "datasets/cifar10.h"
#include <filesystem>
#include <random>
#include <utility>

void CIFAR10::load_data() {
    std::cout << "loading " << dataset_fp_ << std::endl;
    std::ifstream file(dataset_fp_.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Download dataset first!!" << std::endl;
        exit(EXIT_FAILURE);
    }
    int file_size = std::filesystem::file_size(std::filesystem::path(dataset_fp_));

    uint8_t ptr[1];
    auto num_pixels = channels_ * height_ * width_;
    auto *q = new uint8_t[num_pixels];
    for (int i = 0; i < file_size / (num_pixels + 1); i++) {
        std::vector<double> target_one_hot(num_classes_, 0.f);
        file.read((char *)ptr, 1);
        target_one_hot[static_cast<int>(ptr[0])] = 1.f;
        target_pool_.push_back(target_one_hot);

        file.read((char *)q, num_pixels);
        std::vector<double> image(q, &q[num_pixels]);
        data_pool_.push_back(image);
    }

    delete[] q;
    num_batches_ = data_pool_.size() / batch_size_;
    std::cout << "num_batches: " << num_batches_ << std::endl;
    std::cout << "loaded " << data_pool_.size() << " items.." << std::endl;

    file.close();
}
void CIFAR10::load_target() {}
void CIFAR10::normalize_data() {
    for (auto &sample : data_pool_) {
        double *sample_data_ptr = sample.data();
        for (int j = 0; j < channels_ * height_ * width_; j++) {
            sample_data_ptr[j] /= 255.f;
            sample_data_ptr[j] -= 0.5;
            sample_data_ptr[j] /= 0.5;

        }
    }
}
CIFAR10::CIFAR10(
    const string &dataset_fp,
    const string &_label_fp,
    bool shuffle,
    int batch_size,
    int num_classes)
    : Dataset(dataset_fp, "", shuffle, batch_size, num_classes) {

    channels_ = 3;
    height_ = 32;
    width_ = 32;

    CIFAR10::load_data();
    CIFAR10::normalize_data();

    if (shuffle_)
        shuffle_dataset();
    create_shared_space();
}
int CIFAR10::get_num_batches() const { return num_batches_; }
