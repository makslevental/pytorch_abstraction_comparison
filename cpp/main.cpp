#include <iostream>
#include <torch/torch.h>

struct Net : torch::nn::Module {
    torch::nn::Linear linear = nullptr;
    torch::Tensor another_bias;

    Net(int64_t N, int64_t M) {
        linear = register_module("linear", torch::nn::Linear(N, M));
        another_bias = register_parameter("b", torch::randn(M));
    }
    torch::Tensor forward(torch::Tensor input) {
        return linear(input) + another_bias;
    }
};

int main() {
    Net net(4, 5);
    std::cout << net.forward(torch::ones({2, 4})) << std::endl;
}