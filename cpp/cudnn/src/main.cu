#include "CLI11.hpp"
#include "datasets/datasets.h"
#include "network.h"
#include "resnet.cuh"
#include <cassert>
#include <cmath>

#include <cuda_profiler_api.h>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>

int get_tp_count(Tensor<double> *output, Tensor<double> *target);
int arg_max(int batch, int output_size, const double *arr);
int find_one(int batch, int output_size, const double *arr);

int main(int argc, char *argv[]) {
    CLI::App app{"CUDNN Harness"};

    std::string train_dataset_fp = "default";
    std::string train_label_fp = "default";
    app.add_option("--train_dataset_fp", train_dataset_fp, "dataset file path");
    app.add_option("--train_label_fp", train_label_fp, "label file path");

    std::string test_dataset_fp = "default";
    std::string test_label_fp = "default";
    app.add_option("--test_dataset_fp", test_dataset_fp, "dataset file path");
    app.add_option("--test_label_fp", test_label_fp, "label file path");

    CLI11_PARSE(app, argc, argv);

    /* configure the network */
    int batch_size = 2;

    int epochs = 5;
    int monitoring_step = 100;

    double learning_rate = 0.1f;
    double lr_decay = 0.00005f;

    bool load_pretrain = false;
    bool file_save = false;

    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    //    MNIST train_data_loader =
    //        MNIST(train_dataset_fp, train_label_fp, true, batch_size, NUMBER_MNIST_CLASSES);
    //    MNIST test_data_loader =
    //        MNIST(test_dataset_fp, test_label_fp, false, batch_size, NUMBER_MNIST_CLASSES);
    //    STL10 train_data_loader =
    //        STL10(train_dataset_fp, train_label_fp, true, batch_size, NUMBER_STL10_CLASSES);
    //    STL10 test_data_loader =
    //        STL10(test_dataset_fp, test_label_fp, false, batch_size, NUMBER_STL10_CLASSES);
    CIFAR10 train_data_loader =
        CIFAR10(train_dataset_fp, "", true, batch_size, NUMBER_CIFAR10_CLASSES);
    CIFAR10 test_data_loader =
        CIFAR10(test_dataset_fp, "", false, batch_size, NUMBER_CIFAR10_CLASSES);

    //
    //    auto model = make_resnet50();
    //    model->cuda();
    auto model = new Network();
    model->add_layer(new Conv2d("conv1", 20, 5));
    model->add_layer(new Activation("relu1", CUDNN_ACTIVATION_RELU));
    model->add_layer(new Pooling("pool1", 2, 2, 0, CUDNN_POOLING_MAX));
    model->add_layer(new Conv2d("conv2", 50, 5));
    model->add_layer(new Activation("relu2", CUDNN_ACTIVATION_RELU));
    model->add_layer(new Pooling("pool2", 2, 2, 0, CUDNN_POOLING_MAX));
    model->add_layer(new Dense("dense1", 500));
    model->add_layer(new Activation("relu3", CUDNN_ACTIVATION_RELU));
    model->add_layer(new Dense("dense2", 10));
    model->add_layer(new Softmax("softmax"));
    model->cuda();
    checkCudaErrors(cudaDeviceSynchronize());

    if (load_pretrain)
        model->load_pretrain();

    CrossEntropyLoss criterion(batch_size, model->get_cuda_context());
    CrossEntropyLoss criterion1(batch_size, model->get_cuda_context());
    double loss, accuracy;
    int tp_count;
    int sample_count;

    cudaProfilerStart();

    Tensor<double> *train_data, *train_target;
    Tensor<double> *test_data, *test_target;
    Tensor<double> *output;

    std::string nvtx_message;
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "[TRAIN]" << std::endl;
        model->train();
        tp_count = 0;
        sample_count = 0;
        loss = 0;
        train_data_loader.reset();

        for (int batch = 0; batch < train_data_loader.get_num_batches(); batch++) {
            nvtx_message =
                std::string("epoch " + std::to_string(epoch) + " batch " + std::to_string(batch));
            nvtxRangePushA(nvtx_message.c_str());

            std::tie(train_data, train_target) = train_data_loader.get_next_batch();
            train_data->to(cuda);
            train_target->to(cuda);

            output = model->forward(train_data, true);
            tp_count += get_tp_count(output, train_target);
            sample_count += batch_size;
            loss += criterion.loss(output, train_target);
            //            printf("loss %f\n", loss);

            auto gradient = criterion.backward();
            gradient->print("gradient", true, batch_size);
            model->backward(gradient);
            model->update(learning_rate);

            nvtxRangePop();

            if (batch % monitoring_step == 0) {
                //                train_data->print("data", true);
                //                output->print("output", true);
                //                train_target->print("target", true);

                accuracy = 100.f * tp_count / sample_count;
                std::cout << "epoch: " << std::right << std::setw(4) << epoch
                          << ", batch: " << std::right << std::setw(4) << batch
                          << ", avg loss: " << std::left << std::setw(8) << std::scientific
                          << std::setprecision(6) << loss / sample_count
                          << ", accuracy: " << std::fixed << accuracy << "%" << std::endl;
                tp_count = 0;
                sample_count = 0;
                loss = 0;
            }
        }
        std::cout << std::endl;

        if (file_save)
            model->write_file();

        std::cout << "[EVAL]" << std::endl;

        model->eval();
        test_data_loader.reset();

        tp_count = 0;
        sample_count = 0;
        loss = 0;
        for (int batch = 0; batch < test_data_loader.get_num_batches(); batch++) {
            std::string nvtx_message = std::string("batch " + std::to_string(batch));
            nvtxRangePushA(nvtx_message.c_str());

            std::tie(test_data, test_target) = test_data_loader.get_next_batch();
            test_data->to(cuda);
            test_target->to(cuda);

            output = model->forward(test_data);
            tp_count += get_tp_count(output, test_target);
            sample_count += batch_size;
            loss += criterion1.loss(output, test_target);

            nvtxRangePop();
            if (batch % monitoring_step == 0) {
                //                output->print("output", true);
                //                test_target->print("target", true);
                //                test_data->print("data", true);
            }
        }

        accuracy = 100.f * tp_count / sample_count;
        std::cout << "avg loss: " << std::scientific << loss / test_data_loader.len()
                  << ", accuracy: " << std::fixed << accuracy << "%" << std::endl;
        std::cout << std::endl;
    }

    cudaProfilerStop();
    //    model->eval();
    //    test_data_loader.reset();
    //    for (int batch = 0; batch < test_data_loader.get_num_batches(); batch++) {
    //        std::tie(test_data, test_target) = test_data_loader.get_next_batch();
    //        test_data->to(cuda);
    //        test_target->to(cuda);
    //        output = model->forward(test_data);
    //        printf("%d\n", get_tp_count(output, test_target));
    //    }

    std::cout << "Done." << std::endl;

    return 0;
}

int get_tp_count(Tensor<double> *output, Tensor<double> *target) {
    int batch_size = output->get_batch_size();
    int output_size = output->size();

    assert(batch_size == target->get_batch_size());
    assert(output_size == target->size());

    double *h_output, *h_target;
    int idx_output, idx_target;
    int tp_count = 0;

    // get predicts and targets
    h_output = output->to(host);
    h_target = target->to(host);

    // idx_output = idx_target = 0;
    for (int b = 0; b < batch_size; b++) {
        idx_output = arg_max(b, output_size, h_output);
        idx_target = find_one(b, output_size, h_target);
        if (idx_output == idx_target)
            tp_count++;
    }

    return tp_count;
}

int arg_max(int batch, int output_size, const double *arr) {
    int idx_output = 0;
    for (int i = 1; i < NUMBER_MNIST_CLASSES; i++) {
        if (arr[batch * output_size + i] > arr[batch * output_size + idx_output])
            idx_output = i;
    }
    return idx_output;
}

int find_one(int batch, int output_size, const double *arr) {
    for (int i = 0; i < 10; i++) {
        if (abs(arr[batch * output_size + i] - 1) < 1e-10) {
            return i;
        }
    }
    exit(EXIT_FAILURE);
}
