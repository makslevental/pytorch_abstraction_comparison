#ifndef RESNET
#define RESNET

#include <iostream>
#include <stdexcept>
#include <torch/torch.h>
#include <type_traits>
#include <vector>

using namespace torch;
using namespace torch::nn;

/*
based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
*/

Conv2d conv3x3(
    int64_t in_planes,
    int64_t out_planes,
    int64_t stride = 1,
    int64_t groups = 1,
    int64_t dilation = 1) {
    int64_t kernel_size = 3;
    return Conv2d(Conv2dOptions(in_planes, out_planes, kernel_size)
                      .stride(stride)
                      .padding(dilation)
                      .groups(groups)
                      .bias(true)
                      .dilation(dilation));
}

Conv2d conv1x1(int64_t in_planes, int64_t out_planes, int64_t stride = 1, int64_t kernel_size = 1) {
    return Conv2d(Conv2dOptions(in_planes, out_planes, kernel_size).stride(stride).bias(true));
}

struct BasicBlock : Module {
    static const int expansion = 1;
    Conv2d conv1 = nullptr;
    BatchNorm2d bn1 = nullptr;
    Conv2d conv2 = nullptr;
    BatchNorm2d bn2 = nullptr;
    Sequential downsample;
    ReLU relu = nullptr;

    BasicBlock(
        int64_t in_planes,
        int64_t planes,
        int64_t stride = 1,
        Sequential downsample = Sequential(),
        int64_t groups = 1,
        int64_t base_width = 64,
        int64_t dilation = 1) {
        if (groups != 1 || base_width != 64) {
            throw std::runtime_error("BasicBlock only supports groups=1 and base_width=64");
        }
        if (dilation > 1)
            throw std::runtime_error("Dilation > 1 not supported in BasicBlock");

        this->conv1 = conv3x3(in_planes, planes, stride);
        this->bn1 = BatchNorm2d(planes);
        this->relu = ReLU(ReLUOptions(true));
        this->conv2 = conv3x3(planes, planes);
        this->bn2 = BatchNorm2d(planes);
        this->downsample = downsample;
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu", this->relu);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        if (!downsample->is_empty()) {
            register_module("downsample", downsample);
        }
    }

    Tensor forward(torch::Tensor x) {
        at::Tensor identity(x.clone());

        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = this->relu(out);
        out = conv2->forward(out);
        out = bn2->forward(out);
        if (!downsample->is_empty())
            identity = downsample->forward(out);

        out += identity;
        out = this->relu(out);

        return out;
    }
};

struct BottleNeck : Module {
    static const int expansion = 4;
    Conv2d conv1 = nullptr;
    BatchNorm2d bn1 = nullptr;
    Conv2d conv2 = nullptr;
    BatchNorm2d bn2 = nullptr;
    Conv2d conv3 = nullptr;
    BatchNorm2d bn3 = nullptr;
    ReLU relu = nullptr;
    Sequential downsample;

    BottleNeck(
        int64_t in_planes,
        int64_t planes,
        int64_t stride = 1,
        Sequential downsample = Sequential(),
        int64_t groups = 1,
        int64_t base_width = 64,
        int64_t dilation = 1) {
        auto width = int64_t(planes * (base_width / 64.)) * groups;
        this->conv1 = conv1x1(in_planes, width);
        this->bn1 = BatchNorm2d(width);
        this->conv2 = conv3x3(width, width, stride, groups, dilation);
        this->bn2 = BatchNorm2d(width);
        this->conv3 = conv1x1(width, planes * this->expansion);
        this->bn3 = BatchNorm2d(planes * this->expansion);
        this->relu = ReLU(ReLUOptions(true));
        this->downsample = downsample;

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        register_module("relu", this->relu);
        if (!downsample->is_empty()) {
            register_module("downsample", downsample);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        at::Tensor identity(x.clone());

        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = this->relu(out);
        out = conv2->forward(out);
        out = bn2->forward(out);
        out = this->relu(out);
        out = conv3->forward(out);
        out = bn3->forward(out);

        if (!downsample->is_empty())
            identity = downsample->forward(identity);

        out += identity;
        out = this->relu(out);

        return out;
    }
};

template <class Block> struct ResNet : Module {

    int in_planes = 64;
    int dilation = 1;
    int groups;
    int base_width;

    Conv2d conv1 = nullptr;
    BatchNorm2d bn1 = nullptr;
    ReLU relu = nullptr;
    MaxPool2d maxpool = nullptr;
    Sequential layer1;
    Sequential layer2;
    Sequential layer3;
    Sequential layer4;
    AdaptiveAvgPool2d avgpool = nullptr;
    Linear fc = nullptr;

    ResNet(
        int layers[],
        int64_t num_classes = 1000,
        int64_t num_channels = 3,
        int64_t groups = 1,
        int64_t width_per_group = 64,
        std::vector<bool> replace_stride_with_dilation = {}) {
        if (replace_stride_with_dilation.empty())
            replace_stride_with_dilation = {false, false, false};
        if (replace_stride_with_dilation.size() != 3)
            throw std::runtime_error(
                "replace_stride_with_dilation should be None or a 3-element tuple");

        this->groups = groups;
        this->base_width = width_per_group;

        this->conv1 = Conv2d(
            Conv2dOptions(num_channels, this->in_planes, 7).stride(2).padding(3).bias(true));
        this->bn1 = BatchNorm2d(this->in_planes);
        this->relu = ReLU(ReLUOptions(true));
        this->maxpool = MaxPool2d(MaxPool2dOptions(3).stride(2).padding(1));
        this->layer1 = this->make_layer(64, layers[0]);
        this->layer2 = this->make_layer(128, layers[1], 2, replace_stride_with_dilation[0]);
        this->layer3 = this->make_layer(256, layers[2], 2, replace_stride_with_dilation[1]);
        this->layer4 = this->make_layer(512, layers[3], 2, replace_stride_with_dilation[2]);
        this->avgpool = AdaptiveAvgPool2d(1);
        this->fc = Linear(512 * Block::expansion, num_classes);

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu", this->relu);
        register_module("maxpool", this->maxpool);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("avgpool", this->avgpool);
        register_module("fc", fc);
    }

    void initialize_weights() {
        for (auto m : this->modules()) {
            if (m.get()->name() == "torch::nn::Conv2dImpl") {
                for (auto p : m.get()->named_parameters()) {
                    if (p.key() == "weight") {
                        torch::nn::init::kaiming_normal_(
                            p.value(), 0.0, torch::kFanOut, torch::kReLU);
                    }
                }
            } else if (m.get()->name() == "torch::nn::BatchNorm2dImpl") {
                for (auto p : m.get()->named_parameters()) {
                    if (p.key() == "weight") {
                        torch::nn::init::constant_(p.value(), 1);
                    } else if (p.key() == "bias") {
                        torch::nn::init::constant_(p.value(), 0);
                    }
                }
            }
        }
    }

    torch::Tensor forward(torch::Tensor x) {

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = this->relu(x);
        x = this->maxpool(x);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = this->avgpool(x);
        x = x.view({x.sizes()[0], -1});
        x = fc->forward(x);

        return x;
    }

private:
    torch::nn::Sequential
    make_layer(int64_t planes, int64_t blocks, int64_t stride = 1, bool dilate = false) {
        torch::nn::Sequential downsample;
        auto previous_dilation = this->dilation;
        if (dilate) {
            this->dilation *= stride;
            stride = 1;
        }
        if (stride != 1 or this->in_planes != planes * Block::expansion) {
            downsample = Sequential(
                conv1x1(this->in_planes, planes * Block::expansion, stride),
                BatchNorm2d(planes * Block::expansion));
        }
        Sequential layers;
        layers->push_back(Block(
            this->in_planes,
            planes,
            stride,
            downsample,
            this->groups,
            this->base_width,
            previous_dilation));
        this->in_planes = planes * Block::expansion;
        for (int64_t i = 0; i < blocks - 1; i++) {
            layers->push_back(Block(
                this->in_planes,
                planes,
                1,
                Sequential(),
                this->groups,
                this->base_width,
                this->dilation));
        }

        return layers;
    }
};
TORCH_MODULE_IMPL(ResNetBasic, ResNet<BasicBlock>);
TORCH_MODULE_IMPL(ResNetBottleNeck, ResNet<BottleNeck>);

ResNetBasic resnet18(int num_classes = 1000, int64_t num_channels = 3) {
    int layers[] = {2, 2, 2, 2};
    ResNetBasic model(layers, num_classes, num_channels);
    return model;
}

ResNetBasic resnet34(int num_classes = 1000, int64_t num_channels = 3) {
    int layers[] = {3, 4, 6, 3};
    ResNetBasic model(layers, num_classes, num_channels);
    return model;
}

ResNet<BottleNeck> _resnet50(int num_classes = 1000, int64_t num_channels = 3) {
    int layers[] = {3, 4, 6, 3};
    ResNet<BottleNeck> model(layers, num_classes, num_channels);
    return model;
}

ResNetBottleNeck resnet50(int num_classes = 1000, int64_t num_channels = 3) {
    int layers[] = {3, 4, 6, 3};
    ResNetBottleNeck model(layers, num_classes, num_channels);
    return model;
}

ResNetBottleNeck resnet101(int num_classes = 1000, int64_t num_channels = 3) {
    int layers[] = {3, 4, 23, 3};
    ResNetBottleNeck model(layers, num_classes, num_channels);
    return model;
}

ResNetBottleNeck resnet152(int num_classes = 1000, int64_t num_channels = 3) {
    int layers[] = {3, 8, 36, 3};
    ResNetBottleNeck model(layers, num_classes, num_channels);
    return model;
}

#endif