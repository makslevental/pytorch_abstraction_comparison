#include "cuda_helper.h"
#include "cuda_profiling.h"
#include "datasets/datasets.h"
#include "network.h"
#include "resnet.cuh"

#include <cassert>
#include <cmath>
#include <cuda_profiler_api.h>
#include <fstream>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>

template <typename dtype>
int get_tp_count(Tensor<dtype> *output, Tensor<dtype> *target, int num_classes);
template <typename dtype>
int arg_max(int batch, int output_size, int num_classes, const dtype *arr);
template <typename dtype>
int find_one(int batch, int output_size, int num_classes, const dtype *arr);

template <typename dtype>
void train(
    Dataset<dtype> *train_data_loader,
    Dataset<dtype> *test_data_loader,
    int num_classes,
    int epochs,
    int batch_size,
    int monitoring_step,
    double learning_rate,
    std::ostream &output_file) {

    CrossEntropyLoss<dtype> criterion;
    CrossEntropyLoss<dtype> criterion1;

    Network<dtype> *model;
    if (strcmp(std::getenv("MODEL"), "small") == 0) {
        model = new Network<dtype>();
        model->add_layer(new Conv2d<dtype>("conv1", 20, 5));
        model->add_layer(new Activation<dtype>("relu1", CUDNN_ACTIVATION_RELU));
        model->add_layer(new Pooling<dtype>("pool1", 2, 2, 0, CUDNN_POOLING_MAX));
        model->add_layer(new Conv2d<dtype>("conv2", 50, 5));
        model->add_layer(new Activation<dtype>("relu2", CUDNN_ACTIVATION_RELU));
        model->add_layer(new Pooling<dtype>("pool2", 2, 2, 0, CUDNN_POOLING_MAX));
        model->add_layer(new Dense<dtype>("dense1", 500));
        model->add_layer(new Activation<dtype>("relu3", CUDNN_ACTIVATION_RELU));
        model->add_layer(new Dense<dtype>("dense2", num_classes));
        model->add_layer(new Softmax<dtype>("softmax"));
        model->cuda();
    } else if (strcmp(std::getenv("MODEL"), "resnet") == 0) {
        model = make_resnet50<dtype>(num_classes);
        model->cuda();
    } else {
        std::cout << "no model";
        exit(EXIT_FAILURE);
    }

    std::string nvtx_message;
    auto gpu_timer = GPUTimer();
    cudaProfilerStart();

    Tensor<dtype> *train_data, *train_target;
    Tensor<dtype> *test_data, *test_target;
    Tensor<dtype> *output;
    double loss, accuracy, running_loss;
    int tp_count, running_tp_count, running_sample_count, sample_count;
    double total_time;
    double elapsed_time;
    double used_mem = 0, running_used_mem = 0;
    double lr;
    double lr_decay = 0.0000005f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        model->train();
        total_time = loss = accuracy = running_loss = 0;
        running_used_mem = used_mem = elapsed_time = running_sample_count = tp_count =
            running_tp_count = sample_count = 0;
        lr = learning_rate;
        train_data_loader->reset();

        for (int batch = 0; batch < train_data_loader->get_num_batches(); batch++) {
            nvtx_message = std::string(
                "train epoch " + std::to_string(epoch) + " batch " + std::to_string(batch));
            nvtxRangePushA(nvtx_message.c_str());
            nvtxRangePushA("batch load");
            std::tie(train_data, train_target) = train_data_loader->get_next_batch();
            gpu_timer.start();
            train_data->to(cuda);
            train_target->to(cuda);

            nvtxRangePop();

            output = model->forward(train_data);
            loss += criterion.loss(output, train_target);
            model->backward(train_target);
            //            lr *= 1.f / (1.f + lr_decay * batch);
            model->update(lr);

            gpu_timer.stop();

            nvtxRangePop();

            sample_count += batch_size;
            tp_count += get_tp_count<dtype>(output, train_target, num_classes);
            elapsed_time += gpu_timer.elapsed();
            used_mem += get_used_cuda_mem();

            if ((batch + 1) % monitoring_step == 0) {
                std::cout << "batch: " << batch << std::endl;
                accuracy = 100.f * tp_count / sample_count;
                output_file << "[TRAIN] epoch: " << epoch << ", batch: " << batch << std::fixed
                            << std::setprecision(10) << ", avg loss: " << std::left
                            << loss / (float)sample_count << ", accuracy: " << accuracy << "%"
                            << ", avg sample time: " << elapsed_time / sample_count << "ms"
                            << std::defaultfloat << ", avg used mem: " << used_mem / monitoring_step
                            << "mb"
                            << ", avg gpu util: " << get_gpu_utilization() << "%" << std::endl;
                total_time += elapsed_time;
                running_loss += loss;
                running_tp_count += tp_count;
                running_sample_count += sample_count;
                running_used_mem += used_mem;
                used_mem = elapsed_time = tp_count = sample_count = loss = 0;
                output_file.flush();
            }
        }

        output_file << "[TRAIN SUMMARY] avg loss: " << std::left << std::setw(8) << std::fixed
                    << std::setprecision(6) << running_loss / running_sample_count
                    << ", accuracy: " << 100.f * running_tp_count / running_sample_count << "%"
                    << ", avg sample time: " << total_time / running_sample_count << "ms"
                    << std::defaultfloat
                    << ", avg used mem: " << running_used_mem / (running_sample_count / batch_size)
                    << "mb"
                    << ", avg gpu util: " << get_gpu_utilization() << "%" << std::endl;

        output_file.flush();
        model->eval();
        test_data_loader->reset();
        total_time = sample_count = tp_count = loss = 0;
        used_mem = 0;

        for (int batch = 0; batch < test_data_loader->get_num_batches(); batch++) {
            nvtx_message = std::string(
                "eval epoch " + std::to_string(epoch) + " batch " + std::to_string(batch));
            nvtxRangePushA(nvtx_message.c_str());
            nvtxRangePushA("batch load");
            std::tie(test_data, test_target) = test_data_loader->get_next_batch();

            gpu_timer.start();
            test_data->to(cuda);
            test_target->to(cuda);

            nvtxRangePop();
            output = model->forward(test_data);
            loss += criterion1.loss(output, test_target);

            nvtxRangePop();
            tp_count += get_tp_count<dtype>(output, test_target, num_classes);
            used_mem += get_used_cuda_mem();

            gpu_timer.stop();
            sample_count += batch_size;
            total_time += gpu_timer.elapsed();
        }

        accuracy = 100.f * tp_count / sample_count;
        output_file << "[EVAL] avg loss: " << std::setw(4) << loss / sample_count
                    << ", accuracy: " << accuracy << "%"
                    << ", avg sample time: " << total_time / sample_count << "ms"
                    << std::defaultfloat
                    << ", avg used mem: " << used_mem / (sample_count / batch_size) << "mb"
                    << ", avg gpu util: " << get_gpu_utilization() << "%" << std::endl;
        output_file.flush();
    }

    cudaProfilerStop();
    std::cout << "Done." << std::endl;
}

int main(int argc, char *argv[]) {
    int64_t batch_size = std::stoi(std::getenv("BATCH_SIZE"));
    int num_classes;
    int epochs = std::stoi(std::getenv("EPOCHS"));
    int monitoring_step = 20;

    double learning_rate = 1.0 / std::stoi(std::getenv("INV_LEARNING_RATE"));
    double lr_decay = 0.0000005f;

    Dataset<float> *train_data_loader;
    Dataset<float> *test_data_loader;

    std::stringstream ss;
    ss << "profiles/run_cudnn_" << argv[1] << "_" << batch_size << "_" << std::getenv("RESOLUTION")
       << ".csv";
    std::ofstream output_file(ss.str());

    if (strcmp(argv[1], "mnist") == 0) {
        num_classes = NUMBER_MNIST_CLASSES;
        std::cout << "== MNIST training with CUDNN ==" << std::endl;
        train_data_loader = new MNIST<float>(
            "../data/MNIST/raw/train-images-idx3-ubyte",
            "../data/MNIST/raw/train-labels-idx1-ubyte",
            true,
            batch_size,
            NUMBER_MNIST_CLASSES);
        test_data_loader = new MNIST<float>(
            "../data/MNIST/raw/t10k-images-idx3-ubyte",
            "../data/MNIST/raw/t10k-labels-idx1-ubyte",
            false,
            batch_size,
            NUMBER_MNIST_CLASSES);
    } else if (strcmp(argv[1], "stl10") == 0) {
        num_classes = NUMBER_STL10_CLASSES;
        std::cout << "== STL10 training with CUDNN ==" << std::endl;
        train_data_loader = new STL10<float>(
            "../data/stl_10_test_data.npy",
            "../data/stl_10_test_labels.npy",
            true,
            batch_size,
            NUMBER_STL10_CLASSES);
        test_data_loader = new STL10<float>(
            "../data/stl_10_train_data.npy",
            "../data/stl_10_train_labels.npy",
            false,
            batch_size,
            NUMBER_STL10_CLASSES);
    } else if (strcmp(argv[1], "cifar10") == 0) {
        num_classes = NUMBER_CIFAR10_CLASSES;
        std::cout << "== CIFAR10 training with CUDNN ==" << std::endl;
        train_data_loader = new CIFAR10<float>(
            "../data/cifar-10-batches-bin/all_train_data.bin",
            "",
            true,
            batch_size,
            NUMBER_CIFAR10_CLASSES);
        test_data_loader = new CIFAR10<float>(
            "../data/cifar-10-batches-bin/test_batch.bin",
            "",
            false,
            batch_size,
            NUMBER_CIFAR10_CLASSES);
    } else if (strcmp(argv[1], "pascal") == 0) {
        num_classes = NUMBER_PASCAL_CLASSES;
        std::cout << "== PASCAL training with CUDNN ==" << std::endl;
        train_data_loader = new PASCAL<float>(
            "../data/VOCdevkit/VOC2012", Mode::kTrain, true, batch_size, NUMBER_PASCAL_CLASSES);
        test_data_loader = new PASCAL<float>(
            "../data/VOCdevkit/VOC2012", Mode::kVal, false, batch_size, NUMBER_PASCAL_CLASSES);
    } else {
        exit(EXIT_FAILURE);
    }
    train<float>(
        train_data_loader,
        test_data_loader,
        num_classes,
        epochs,
        batch_size,
        monitoring_step,
        learning_rate,
        output_file);

    return 0;
}

template <typename dtype>
int get_tp_count(Tensor<dtype> *output, Tensor<dtype> *target, int num_classes) {
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
        idx_output = arg_max<dtype>(b, output_size, num_classes, h_output);
        idx_target = find_one<dtype>(b, output_size, num_classes, h_target);
        if (idx_output == idx_target)
            tp_count++;
    }

    return tp_count;
}

template <typename dtype>
int arg_max(int batch, int output_size, int num_classes, const dtype *arr) {
    int idx_output = 0;
    for (int i = 1; i < num_classes; i++) {
        if (arr[batch * output_size + i] > arr[batch * output_size + idx_output])
            idx_output = i;
    }
    return idx_output;
}

template <typename dtype>
int find_one(int batch, int output_size, int num_classes, const dtype *arr) {
    for (int i = 0; i < num_classes; i++) {
        if (abs(arr[batch * output_size + i] - 1) < 1e-10) {
            return i;
        }
    }
    std::cout << "no one found\n";
    exit(EXIT_FAILURE);
}
