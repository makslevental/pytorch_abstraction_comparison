#ifndef UTIL
#define UTIL

#include <inttypes.h>
#include <iostream>
#include <regex>
#include <set>
#include <torch/torch.h>

void tabs(size_t num) {
    for (size_t i = 0; i < num; i++) {
        std::cout << "  ";
    }
}

static const std::set<std::string> BLACKLIST_PARAMS = {
    "ResNet<BottleNeck>", "SequentialImpl", "BottleNeck", "BasicBlock"};

void print_modules(const std::shared_ptr<torch::nn::Module> &module, size_t level = 0) {
    module->pretty_print(std::cout);
    std::cout << "\n";
    for (const auto &child : module->named_children()) {
        tabs(level + 1);
        std::cout << "(" << child.key() << "): ";
        print_modules(child.value(), level + 1);
    }
}

#endif