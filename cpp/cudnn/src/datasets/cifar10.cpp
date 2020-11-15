//
// Created by Maksim Levental on 11/13/20.
//

#include "datasets/cifar10.h"
#include <filesystem>
#include <random>
#include <utility>

template <typename dtype> void CIFAR10<dtype>::load_data() {
    std::cout << "loading " << this->dataset_fp_ << std::endl;
    std::ifstream file(this->dataset_fp_.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Download dataset first!!" << std::endl;
        exit(EXIT_FAILURE);
    }
    int file_size = std::filesystem::file_size(std::filesystem::path(this->dataset_fp_));

    uint8_t ptr[1];
    auto num_pixels = this->channels_ * this->height_ * this->width_;
    auto *q = new uint8_t[num_pixels];
    for (int i = 0; i < file_size / (num_pixels + 1); i++) {
        std::vector<double> target_one_hot(this->num_classes_, 0.f);
        file.read((char *)ptr, 1);
        target_one_hot[static_cast<int>(ptr[0])] = 1.f;
        this->target_pool_.push_back(target_one_hot);

        file.read((char *)q, num_pixels);
        std::vector<double> image(q, &q[num_pixels]);
        this->data_pool_.push_back(image);
    }

    delete[] q;
    this->num_batches_ = this->data_pool_.size() / this->batch_size_;
    std::cout << "num_batches: " << this->num_batches_ << std::endl;
    std::cout << "loaded " << this->data_pool_.size() << " items.." << std::endl;

    file.close();
}
template <typename dtype> void CIFAR10<dtype>::load_target() {}
template <typename dtype> void CIFAR10<dtype>::normalize_data() {
    for (auto &sample : this->data_pool_) {
        double *sample_data_ptr = sample.data();
        for (int j = 0; j < this->channels_ * this->height_ * this->width_; j++) {
            sample_data_ptr[j] /= 255.f;
            sample_data_ptr[j] -= 0.5;
            sample_data_ptr[j] /= 0.5;
        }
    }
}
template <typename dtype>
CIFAR10<dtype>::CIFAR10(
    const string &dataset_fp,
    const string &_label_fp,
    bool shuffle,
    int batch_size,
    int num_classes)
    : Dataset<dtype>(dataset_fp, "", shuffle, batch_size, num_classes) {

    this->channels_ = 3;
    this->height_ = 32;
    this->width_ = 32;

    CIFAR10::load_data();
    CIFAR10::normalize_data();

    if (this->shuffle_)
        this->shuffle_dataset();
    this->create_shared_space();
}
template <typename dtype> int CIFAR10<dtype>::get_num_batches() const { return this->num_batches_; }

template class CIFAR10<float>;
template class CIFAR10<double>;
