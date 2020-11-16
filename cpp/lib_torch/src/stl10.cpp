//
// Created by Maksim Levental on 11/15/20.
//

#include "stl10.h"
#include "npy.h"

#include <vector>

namespace {
    constexpr uint32_t channels_ = 3;
    constexpr uint32_t height_ = 96;
    constexpr uint32_t width_ = 96;

    std::pair<torch::Tensor, torch::Tensor>
    read_data(const std::string &dataset_fp_, const std::string &label_fp) {
        std::vector<unsigned long> data_shape;
        bool fortran_order;
        std::vector<uint8_t> data_buffer;
        npy::LoadArrayFromNumpy(dataset_fp_, data_shape, fortran_order, data_buffer);
        // n x c x h x w
        assert(data_shape.size() == 4);
        int num_samples = data_shape[0];
        assert(channels_ == data_shape[1]);
        assert(height_ == data_shape[2]);
        assert(width_ == data_shape[3]);

        auto num_pixels = channels_ * height_ * width_;
        auto targets = torch::empty(num_samples, torch::kByte);
        auto images = torch::empty({num_samples, channels_, height_, width_}, torch::kByte);

        for (uint32_t i = 0; i != num_samples; ++i) {
            uint32_t start_index = i * num_pixels;

            uint32_t image_start = start_index + 1;
            uint32_t image_end = image_start + num_pixels;
            std::copy(
                data_buffer.begin() + image_start,
                data_buffer.begin() + image_end,
                reinterpret_cast<char *>(images[i].data_ptr()));
        }

        std::vector<unsigned long> label_shape;
        std::vector<uint8_t> label_buffer;

        npy::LoadArrayFromNumpy(label_fp, label_shape, fortran_order, label_buffer);
        assert(label_shape.size() == 1);
        int n_targets = label_shape[0];
        assert(n_targets == num_samples);

        for (int i = 0; i < n_targets; i++) {
            targets[i] = label_buffer[i];
        }

        return {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
    }
} // namespace

STL10::STL10(const std::string &data_fp, const std::string &label_fp) {
    auto data = read_data(data_fp, label_fp);

    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}
torch::data::Example<> STL10::get(size_t index) { return {images_[index], targets_[index]}; }
torch::optional<size_t> STL10::size() const { return images_.size(0); }
[[maybe_unused]] const torch::Tensor &STL10::images() const { return images_; }
[[maybe_unused]] const torch::Tensor &STL10::targets() const { return targets_; }
