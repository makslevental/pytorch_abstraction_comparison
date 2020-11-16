//
// Created by Maksim Levental on 11/15/20.
//

#ifndef PYTORCH_ABSTRACTION_STL10_H
#define PYTORCH_ABSTRACTION_STL10_H

#define NUMBER_STL10_CLASSES 10

#include <cstddef>
#include <fstream>
#include <string>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

class STL10 : public torch::data::datasets::Dataset<STL10> {
public:
    explicit STL10(const std::string &data_fp, const std::string &label_fp);

    torch::data::Example<> get(size_t index) override;
    [[nodiscard]] torch::optional<size_t> size() const override;
    [[maybe_unused]] [[maybe_unused]] [[nodiscard]] const torch::Tensor &images() const;
    [[maybe_unused]] [[maybe_unused]] [[maybe_unused]] [[nodiscard]] const torch::Tensor &
    targets() const;

private:
    torch::Tensor images_;
    torch::Tensor targets_;
};
#endif // PYTORCH_ABSTRACTION_STL10_H
