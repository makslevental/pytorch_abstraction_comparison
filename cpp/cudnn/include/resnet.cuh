//
// Created by Maksim Levental on 10/29/20.
//

#include "layers.cuh"
#include "network.h"

Network make_resnet50() {
    Network model;

    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 64, /*kernel*/ 7, /*stride*/ 2, /*padding*/ 3));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(new Activation("relu1", CUDNN_ACTIVATION_RELU));
    model.add_layer(
        new Pooling("pool1", /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1, CUDNN_POOLING_MAX));

    // layer 1
    // bottleneck 1
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 64, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 64, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2D("conv3", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    // downsample
    model.add_layer(
        new Conv2D("0", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("1"));

    // bottleneck 2
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 64, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 64, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2D("conv3", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 3
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 64, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 64, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2D("conv3", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // layer 2
    // bottleneck 1
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2D("conv3", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    // downsample
    model.add_layer(
        new Conv2D("0", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 2, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("1"));

    // bottleneck 2
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2D("conv3", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 3
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2D("conv3", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 4
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2D("conv3", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // layer 3
    // bottleneck 1
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2D(
        "conv3",
        /*out_channels*/ 1024,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    // downsample
    model.add_layer(
        new Conv2D("0", /*out_channels*/ 1024, /*kernel*/ 1, /*stride*/ 2, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("1"));

    // bottleneck 2
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2D(
        "conv3",
        /*out_channels*/ 1024,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 3
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2D(
        "conv3",
        /*out_channels*/ 1024,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 4
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2D(
        "conv3",
        /*out_channels*/ 1024,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 5
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2D(
        "conv3",
        /*out_channels*/ 1024,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // layer 4
    // bottleneck 1
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 512, /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2D(
        "conv3",
        /*out_channels*/ 2048,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    // downsample
    model.add_layer(
        new Conv2D("0", /*out_channels*/ 2048, /*kernel*/ 1, /*stride*/ 2, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("1"));

    // bottleneck 2
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 512, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2D(
        "conv3",
        /*out_channels*/ 2048,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 3
    model.add_layer(
        new Conv2D("conv1", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2D("conv2", /*out_channels*/ 512, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2D(
        "conv3",
        /*out_channels*/ 2048,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    model.add_layer(
        new Pooling("pool1", /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0, CUDNN_POOLING_MAX));
    model.add_layer(new Dense("dense1", 10));
    model.add_layer(new Softmax("softmax"));
    return model;
}