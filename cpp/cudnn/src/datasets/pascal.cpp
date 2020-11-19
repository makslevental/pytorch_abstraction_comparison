//
// Created by Maksim Levental on 11/10/20.
//

#include <datasets/pascal.h>

#include "npy.h"
#include "progress_bar.h"
#include <cassert>
#include <utility>

template <typename dtype> void PASCAL<dtype>::load_data() {
    // n x c x h x w

    int n_samples = pascal_generic_.len();
    this->channels_ = pascal_generic_.channels_, this->height_ = pascal_generic_.height_,
    this->width_ = pascal_generic_.width_;

    progresscpp::ProgressBar progressBar(pascal_generic_.len(), 70);
    for (uint32_t i = 0; i != pascal_generic_.len(); ++i) {
        auto [img, label] = pascal_generic_[i];
        std::vector<dtype> image(img.begin<uint8_t>(), img.end<uint8_t>());
        this->data_pool_.push_back(image);

        std::vector<dtype> target_batch(this->num_classes_, 0.f);
        target_batch[static_cast<int>(label)] = 1.f;
        this->target_pool_.push_back(target_batch);
        ++progressBar;
        progressBar.display();
    }
    progressBar.done();

    this->num_batches_ = (int)(n_samples / this->batch_size_);
    std::cout << "num_batches: " << this->num_batches_ << std::endl;
    std::cout << "loaded " << this->data_pool_.size() << " items.." << std::endl;
}

template <typename dtype> void PASCAL<dtype>::load_target() {}

template <typename dtype> void PASCAL<dtype>::normalize_data() {
    progresscpp::ProgressBar progressBar(pascal_generic_.len(), 70);
    for (auto &sample : this->data_pool_) {

        std::transform(
            sample.begin(), sample.end(), sample.begin(), [](int i) { return i / 255.f; });
        ++progressBar;
        progressBar.display();
    }
    progressBar.done();
}

template <typename dtype>
PASCAL<dtype>::PASCAL(
    const string &root_dir, Mode mode, bool shuffle, int batch_size, int num_classes)
    : Dataset<dtype>(root_dir, root_dir, shuffle, batch_size, num_classes),
      pascal_generic_(root_dir, mode == Mode::kTrain ? Split::train_ : Split::val) {

    PASCAL::load_data();
    PASCAL::normalize_data();

    if (this->shuffle_)
        this->shuffle_dataset();
    this->create_shared_space();
}

template <typename dtype> int PASCAL<dtype>::get_num_batches() const { return this->num_batches_; }

template class PASCAL<float>;
template class PASCAL<double>;
