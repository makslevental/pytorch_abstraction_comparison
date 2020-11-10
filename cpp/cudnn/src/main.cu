#include "CLI11.hpp"
#include <cmath>
#include <mnist.h>
#include <network.h>
#include <resnet.cuh>

#include <cuda_profiler_api.h>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>

int get_accuracy(Tensor<float> *output, Tensor<float> *target);
int arg_max(int batch, int output_size, const float *arr);
int find_one(int batch, int output_size, const float *arr);

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
    int batch_size = 32;

    int epochs = 100;
    int monitoring_step = 100;

    double learning_rate = 0.1f;
    double lr_decay = 0.00005f;

    bool load_pretrain = false;
    bool file_save = false;

    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    MNIST train_data_loader =
        MNIST(train_dataset_fp, train_label_fp, true, batch_size, NUMBER_MNIST_CLASSES);
    MNIST test_data_loader =
        MNIST(test_dataset_fp, test_label_fp, false, batch_size, NUMBER_MNIST_CLASSES);

    //    Tensor<float> *train_data, *train_target;
    //    for (int batch = 0; batch < 10; batch++) {
    //        std::tie(train_data, train_target) = train_data_loader.get_next_batch();
    //        train_data->print("train_data", true, batch_size);
    //        train_target->print("train_target", true, batch_size);
    //    }
    //    exit(EXIT_FAILURE);
    CrossEntropyLoss criterion;
    CrossEntropyLoss criterion1;
    double loss, accuracy;
    int tp_count;

    auto model = make_resnet50();
    model->cuda();
    //    Network model;
    //    model.add_layer(new Conv2d("conv1", 20, 5));
    //    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    //    model.add_layer(new Pooling("pool", 2, 2, 0, CUDNN_POOLING_MAX));
    //    model.add_layer(new Conv2d("conv2", 50, 5));
    //    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    //    model.add_layer(new Pooling("pool", 2, 2, 0, CUDNN_POOLING_MAX));
    //    model.add_layer(new Dense("dense1", 500));
    //    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    //    model.add_layer(new Dense("dense2", 10));
    //    model.add_layer(new Softmax("softmax"));
    //    model.cuda();

    if (load_pretrain)
        model->load_pretrain();

    cudaProfilerStart();

    Tensor<float> *train_data, *train_target;
    Tensor<float> *test_data, *test_target;
    Tensor<float> *output;

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "[TRAIN]" << std::endl;
        model->train();
        tp_count = 0;
        train_data_loader.reset();

        for (int batch = 0; batch < train_data_loader.get_num_batches(); batch++) {
            std::string nvtx_message =
                std::string("epoch " + std::to_string(epoch) + " batch " + std::to_string(batch));
            nvtxRangePushA(nvtx_message.c_str());

            std::tie(train_data, train_target) = train_data_loader.get_next_batch();
            train_data->to(cuda);
//            train_data->print("train_data", true);
            train_target->to(cuda);
//            train_target->print("train_target", true);

            output = model->forward(train_data);
            tp_count += get_accuracy(output, train_target);

            model->backward(train_target);
            model->update(learning_rate);

            nvtxRangePop();

            if (batch % monitoring_step == 0) {
                //                train_data->print("data", true, batch_size);
                //                output->print("output", true, batch_size);
                //                train_target->print("target", true, batch_size);

                loss = criterion.loss(output, train_target);
                accuracy = 100.f * tp_count / monitoring_step / batch_size;
                std::cout << "epoch: " << std::right << std::setw(4) << epoch
                          << ", batch: " << std::right << std::setw(4) << batch
                          << ", loss: " << std::left << std::setw(8) << std::fixed
                          << std::setprecision(6) << loss << ", accuracy: " << accuracy << "%"
                          << std::endl;
                tp_count = 0;
            }
        }
        std::cout << std::endl;

        if (file_save)
            model->write_file();

        std::cout << "[EVAL]" << std::endl;

        model->eval();
        test_data_loader.reset();

        tp_count = 0;
        loss = 0;
        for (int batch = 0; batch < test_data_loader.get_num_batches(); batch++) {
            std::string nvtx_message = std::string("batch " + std::to_string(batch));
            nvtxRangePushA(nvtx_message.c_str());

            std::tie(test_data, test_target) = test_data_loader.get_next_batch();
            test_data->to(cuda);
            test_target->to(cuda);

            output = model->forward(test_data);
            tp_count += get_accuracy(output, test_target);
            loss += criterion1.loss(output, test_target);

            nvtxRangePop();
            if (batch % monitoring_step == 0) {
                //                test_data->print("data", true, batch_size);
                //                output->print("output", true, batch_size);
                //                test_target->print("target", true, batch_size);
            }
        }

        accuracy = 100.f * tp_count / test_data_loader.get_num_batches() / batch_size;
        std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%"
                  << std::endl;
        std::cout << std::endl;
    }

    cudaProfilerStop();
    std::cout << "Done." << std::endl;

    return 0;
}

int get_accuracy(Tensor<float> *output, Tensor<float> *target) {
    int batch_size = output->get_batch_size();
    int output_size = output->size();

    assert(batch_size == target->get_batch_size());
    assert(output_size == target->size());

    float *h_output, *h_target;
    int idx_output, idx_target;
    int hit_count = 0;

    // get predicts and targets
    h_output = output->to(host);
    h_target = target->to(host);

    // idx_output = idx_target = 0;
    for (int b = 0; b < batch_size; b++) {
        idx_output = arg_max(b, output_size, h_output);
        idx_target = find_one(b, output_size, h_target);
        if (idx_output == idx_target)
            hit_count++;
    }

    return hit_count;
}

int arg_max(int batch, int output_size, const float *arr) {
    int idx_output = 0;
    for (int i = 1; i < NUMBER_MNIST_CLASSES; i++) {
        if (arr[batch * output_size + i] > arr[batch * output_size + idx_output])
            idx_output = i;
    }
    return idx_output;
}

int find_one(int batch, int output_size, const float *arr) {
    for (int i = 0; i < 10; i++) {
        if (abs(arr[batch * output_size + i] - 1) < 1e-10) {
            return i;
        }
    }
    exit(EXIT_FAILURE);
}