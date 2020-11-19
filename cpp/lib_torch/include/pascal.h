//
// Created by Maksim Levental on 11/18/20.
//

#ifndef PYTORCH_ABSTRACTION_PASCAL_H
#define PYTORCH_ABSTRACTION_PASCAL_H

#include "pascal_generic.h"

#include <cstddef>
#include <fstream>
#include <string>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#define NUMBER_PASCAL_CLASSES 20

class PASCAL : public torch::data::datasets::Dataset<PASCAL> {
public:
    enum Mode { kTrain, kVal };
    PASCAL(const std::string &root, Mode mode);

    torch::data::Example<> get(size_t index) override;
    [[nodiscard]] torch::optional<size_t> size() const override;
    [[maybe_unused]] [[maybe_unused]] [[nodiscard]] const torch::Tensor &images() const;
    [[maybe_unused]] [[maybe_unused]] [[maybe_unused]] [[nodiscard]] const torch::Tensor &
    targets() const;

private:
    torch::Tensor images_;
    torch::Tensor targets_;
    PASCALGeneric pascal_generic_;
};

#endif // PYTORCH_ABSTRACTION_PASCAL_H
