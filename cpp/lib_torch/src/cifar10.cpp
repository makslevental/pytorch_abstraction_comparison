//
// Created by Maksim Levental on 11/15/20.
//

#include "cifar10.h"
#include <filesystem>


namespace {
    constexpr uint32_t channels_ = 3;
    constexpr uint32_t height_ = 32;
    constexpr uint32_t width_ = 32;

    std::pair<torch::Tensor, torch::Tensor> read_data(const std::string &fp) {
        std::vector<char> data_buffer;
        std::ifstream data(fp, std::ios::binary);
        TORCH_CHECK(data, "Error opening data file at", fp);

        int file_size = std::filesystem::file_size(std::filesystem::path(fp));

        auto num_pixels = channels_ * height_ * width_;
        auto bytes_per_row = num_pixels + 1;
        auto num_samples = file_size / (num_pixels + 1);

        data_buffer.insert(data_buffer.end(), std::istreambuf_iterator<char>(data), {});
        auto targets = torch::empty(num_samples, torch::kByte);
        auto images = torch::empty({num_samples, channels_, height_, width_}, torch::kByte);

        for (uint32_t i = 0; i != num_samples; ++i) {
            uint32_t start_index = i * bytes_per_row;
            targets[i] = data_buffer[start_index];

            uint32_t image_start = start_index + 1;
            uint32_t image_end = image_start + num_pixels;
            std::copy(
                data_buffer.begin() + image_start,
                data_buffer.begin() + image_end,
                reinterpret_cast<char *>(images[i].data_ptr()));
        }

        return {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
    }
} // namespace

CIFAR10::CIFAR10(const std::string &fp) {
    auto data = read_data(fp);

    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}

torch::data::Example<> CIFAR10::get(size_t index) { return {images_[index], targets_[index]}; }

torch::optional<size_t> CIFAR10::size() const { return images_.size(0); }

[[maybe_unused]] const torch::Tensor &CIFAR10::images() const { return images_; }

[[maybe_unused]] const torch::Tensor &CIFAR10::targets() const { return targets_; }
