//
// Created by Maksim Levental on 11/18/20.
//

#include "pascal.h"
#include "pascal_generic.h"
#include <string>
#include <torch/data/example.h>
#include <torch/types.h>

PASCAL::PASCAL(const std::string &root, Mode mode)
    : pascal_generic_(root, mode == Mode::kTrain ? Split::train_ : Split::val) {
    auto targets = torch::empty(pascal_generic_.len(), torch::kByte);
    auto images = torch::empty(
        {pascal_generic_.len(),
         pascal_generic_.channels_,
         pascal_generic_.height_,
         pascal_generic_.width_},
        torch::kByte);

    for (uint32_t i = 0; i != pascal_generic_.len(); ++i) {
        auto [img, label] = pascal_generic_[i];
        std::copy(
            img.begin<char>(), img.end<char>(), reinterpret_cast<char *>(images[i].data_ptr()));
        targets[i] = label;
    }
    images_ = std::move(images.to(torch::kFloat32).div_(255));
    targets_ = std::move(targets.to(torch::kInt64));
}
torch::data::Example<> PASCAL::get(size_t index) { return {images_[index], targets_[index]}; }
torch::optional<size_t> PASCAL::size() const { return images_.size(0); }
const torch::Tensor &PASCAL::images() const { return images_; }
const torch::Tensor &PASCAL::targets() const { return targets_; }
