//#include "CLI11.hpp"
#include "datasets/datasets.h"
#include "network.h"
#include "resnet.cuh"
#include <cassert>
#include <cmath>

#include <cuda_profiler_api.h>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>

template <typename dtype> int get_tp_count(Tensor<dtype> *output, Tensor<dtype> *target);
template <typename dtype> int arg_max(int batch, int output_size, const dtype *arr);
template <typename dtype> int find_one(int batch, int output_size, const dtype *arr);

template <typename dtype> void train() {
    int batch_size = 32;

    int epochs = 100;
    int monitoring_step = 20;

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
    auto train_data_loader = CIFAR10<dtype>(
        "/home/maksim/dev_projects/pytorch_abstraction_comparison/data/cifar-10-batches-bin/"
        "all_train_data.bin",
        "",
        true,
        batch_size,
        NUMBER_CIFAR10_CLASSES);
    auto test_data_loader = CIFAR10<dtype>(
        "/home/maksim/dev_projects/pytorch_abstraction_comparison/data/cifar-10-batches-bin/"
        "test_batch.bin",
        "",
        false,
        batch_size,
        NUMBER_CIFAR10_CLASSES);

    CrossEntropyLoss<dtype> criterion;
    CrossEntropyLoss<dtype> criterion1;
    double loss, accuracy;
    int tp_count;
    int sample_count;

    auto model = make_resnet50<dtype>();
    model->cuda();
    //    auto model = new Network<double>();
    //    model->add_layer(new Conv2d<double>("conv1", 20, 5));
    //    model->add_layer(new Activation<double>("relu1", CUDNN_ACTIVATION_RELU));
    //    model->add_layer(new Pooling<double>("pool1", 2, 2, 0, CUDNN_POOLING_MAX));
    //    model->add_layer(new Conv2d<double>("conv2", 50, 5));
    //    model->add_layer(new Activation<double>("relu2", CUDNN_ACTIVATION_RELU));
    //    model->add_layer(new Pooling<double>("pool2", 2, 2, 0, CUDNN_POOLING_MAX));
    //    model->add_layer(new Dense<double>("dense1", 500));
    //    model->add_layer(new Activation<double>("relu3", CUDNN_ACTIVATION_RELU));
    //    model->add_layer(new Dense<double>("dense2", 10));
    //    model->add_layer(new Softmax<double>("softmax"));
    //    model->cuda();
    checkCudaErrors(cudaDeviceSynchronize());

    if (load_pretrain)
        model->load_pretrain();

    cudaProfilerStart();

    Tensor<dtype> *train_data, *train_target;
    Tensor<dtype> *test_data, *test_target;
    Tensor<dtype> *output;

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
            checkCudaErrors(cudaDeviceSynchronize());
            train_data->to(cuda);
            train_target->to(cuda);

            output = model->forward(train_data);
            tp_count += get_tp_count<dtype>(output, train_target);
            loss += criterion.loss(output, train_target);
            sample_count += batch_size;

            model->backward(train_target);
            model->update(learning_rate);

            nvtxRangePop();
            checkCudaErrors(cudaDeviceSynchronize());

            if (batch % monitoring_step == 0) {
                //                train_data->print("data", true, batch_size);
                //                output->print("output", true, batch_size);
                //                train_target->print("target", true, batch_size);

                accuracy = 100.f * tp_count / sample_count;
                std::cout << "epoch: " << std::right << std::setw(4) << epoch
                          << ", batch: " << std::right << std::setw(4) << batch
                          << ", avg loss: " << std::left << std::setw(8) << std::fixed
                          << std::setprecision(6) << loss / (float)sample_count
                          << ", accuracy: " << accuracy << "%";
                std::cout << std::endl;
                tp_count = 0;
                sample_count = 0;
                loss = 0;
            }
        }

        //        accuracy = 100.f * tp_count / sample_count;
        //        std::cout << "avg loss: " << std::left << std::setw(8) << std::fixed <<
        //        std::setprecision(6)
        //                  << loss / (float)sample_count << ", accuracy: " << accuracy << "%";
        //        std::cout << std::endl;
        //        tp_count = 0;
        //        sample_count = 0;
        //        loss = 0;

        if (file_save)
            model->write_file();

        std::cout << "[EVAL]" << std::endl;

        model->eval();
        test_data_loader.reset();

        for (int batch = 0; batch < test_data_loader.get_num_batches(); batch++) {
            std::string nvtx_message = std::string("batch " + std::to_string(batch));
            nvtxRangePushA(nvtx_message.c_str());

            std::tie(test_data, test_target) = test_data_loader.get_next_batch();
            test_data->to(cuda);
            test_target->to(cuda);

            output = model->forward(test_data);
            tp_count += get_tp_count<dtype>(output, test_target);
            sample_count += batch_size;
            loss += criterion1.loss(output, test_target);

            nvtxRangePop();
            if (batch % monitoring_step == 0) {
                //                test_data->print("data", true, batch_size);
                //                output->print("output", true, batch_size);
                //                test_target->print("target", true, batch_size);
            }
        }

        accuracy = 100.f * tp_count / sample_count;
        std::cout << "avg loss: " << std::setw(4) << loss / (float)sample_count
                  << ", accuracy: " << accuracy << "%" << std::endl;
        std::cout << std::endl;
    }

    cudaProfilerStop();
    std::cout << "Done." << std::endl;
}

int main(int argc, char *argv[]) {
    //    CLI::App app{"CUDNN Harness"};
    //
    //    std::string train_dataset_fp = "default";
    //    std::string train_label_fp = "default";
    //    app.add_option("--train_dataset_fp", train_dataset_fp, "dataset file path");
    //    app.add_option("--train_label_fp", train_label_fp, "label file path");
    //
    //    std::string test_dataset_fp = "default";
    //    std::string test_label_fp = "default";
    //    app.add_option("--test_dataset_fp", test_dataset_fp, "dataset file path");
    //    app.add_option("--test_label_fp", test_label_fp, "label file path");
    //
    //    CLI11_PARSE(app, argc, argv);

    /* configure the network */
    train<float>();

    return 0;
}

template <typename dtype> int get_tp_count(Tensor<dtype> *output, Tensor<dtype> *target) {
    int batch_size = output->get_batch_size();
    int output_size = output->size();

    assert(batch_size == target->get_batch_size());
    assert(output_size == target->size());

    dtype *h_output, *h_target;
    int idx_output, idx_target;
    int tp_count = 0;

    // get predicts and targets
    h_output = output->to(host);
    h_target = target->to(host);

    // idx_output = idx_target = 0;
    for (int b = 0; b < batch_size; b++) {
        idx_output = arg_max<dtype>(b, output_size, h_output);
        idx_target = find_one<dtype>(b, output_size, h_target);
        if (idx_output == idx_target)
            tp_count++;
    }

    return tp_count;
}

template <typename dtype> int arg_max(int batch, int output_size, const dtype *arr) {
    int idx_output = 0;
    for (int i = 1; i < NUMBER_MNIST_CLASSES; i++) {
        if (arr[batch * output_size + i] > arr[batch * output_size + idx_output])
            idx_output = i;
    }
    return idx_output;
}

template <typename dtype> int find_one(int batch, int output_size, const dtype *arr) {
    for (int i = 0; i < 10; i++) {
        if (abs(arr[batch * output_size + i] - 1) < 1e-10) {
            return i;
        }
    }
    exit(EXIT_FAILURE);
}
