#include "cifar10.h"
#include "gputimer.h"
#include "helper.h"
#include "resnet.h"
#include "transform.h"

#include <cuda_profiler_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <torch/torch.h>
#include <tuple>

using namespace torch;
using torch::data::datasets::MNIST;
using transform::RandomHorizontalFlip;

template <typename DatasetType>
using TrainDataset = data::datasets::MapDataset<
    data::datasets::MapDataset<DatasetType, RandomHorizontalFlip>,
    data::transforms::Stack<data::Example<>>>;
template <typename DatasetType>
using TestDataset =
    data::datasets::MapDataset<DatasetType, data::transforms::Stack<data::Example<>>>;

template <typename DatasetType>
void train(
    ResNetBottleNeck model,
    TestDataset<DatasetType> train_dataset,
    TestDataset<DatasetType> test_dataset,
    int64_t batch_size,
    int monitoring_step,
    double learning_rate,
    size_t num_epochs,
    torch::Device device,
    std::ofstream &output_file) {

    auto num_train_samples = train_dataset.size().value();
    auto num_test_samples = test_dataset.size().value();
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    torch::optim::SGD optimizer(model->parameters(), learning_rate);
    std::cout << std::fixed << std::setprecision(4);

    std::string nvtx_message;
    auto gpu_timer = GpuTimer();
    cudaProfilerStart();

    double loss_val, accuracy, running_loss;
    int tp_count, running_tp_count, running_sample_count, sample_count;
    double total_time;
    double elapsed_time;
    int batch_n = 0;

    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        model->train();

        total_time = loss_val = accuracy = running_loss = 0;
        elapsed_time = running_sample_count = tp_count = running_tp_count = sample_count = 0;
        batch_n = 0;

        for (auto &batch : *train_loader) {
            nvtx_message = std::string(
                "train epoch " + std::to_string(epoch) + " batch " + std::to_string(batch_n));
            nvtxRangePushA(nvtx_message.c_str());
            nvtxRangePushA("batch load");

            gpu_timer.start();

            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);
            loss_val += loss.template item<double>();
            auto prediction = output.argmax(1);
            tp_count += prediction.eq(target).sum().template item<int64_t>();

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            nvtxRangePop();

            sample_count += batch_size;
            gpu_timer.stop();
            elapsed_time += gpu_timer.elapsed();

            if (batch_n % monitoring_step == 0) {
                std::cout << "batch: " << batch_n << std::endl;
                accuracy = 100.f * tp_count / sample_count;
                output_file << "[TRAIN] epoch: " << std::right << std::setw(4) << epoch
                            << ", batch: " << std::right << std::setw(4) << batch_n
                            << ", avg loss: " << std::left << std::setw(8) << std::fixed
                            << std::setprecision(6) << loss_val / (float)sample_count
                            << ", accuracy: " << accuracy << "%"
                            << ", avg sample time: " << elapsed_time / sample_count << "ms"
                            << ", used mem: " << get_used_cuda_mem() << "mb" << std::endl;
                total_time += elapsed_time;
                running_loss += loss_val;
                running_tp_count += tp_count;
                running_sample_count += sample_count;
                elapsed_time = tp_count = sample_count = loss_val = 0;
            }
            batch_n++;
        }
        output_file << "[TRAIN] avg loss: " << std::left << std::setw(8) << std::fixed
                    << std::setprecision(6) << running_loss / running_sample_count
                    << ", accuracy: " << 100.f * running_tp_count / running_sample_count << "%"
                    << ", avg sample time: " << total_time / running_sample_count << "ms"
                    << std::endl;

        {
            model->eval();
            torch::NoGradGuard no_grad;

            total_time = sample_count = tp_count = loss_val = 0;

            for (const auto &batch : *test_loader) {
                nvtx_message = std::string(
                    "eval epoch " + std::to_string(epoch) + " batch " + std::to_string(batch_n));
                nvtxRangePushA(nvtx_message.c_str());
                nvtxRangePushA("batch load");

                gpu_timer.start();

                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                nvtxRangePop();

                auto output = model->forward(data);

                gpu_timer.stop();

                nvtxRangePop();

                auto loss = torch::nn::functional::cross_entropy(output, target);
                loss_val += loss.template item<double>();

                auto prediction = output.argmax(1);
                tp_count += prediction.eq(target).sum().template item<int64_t>();
                sample_count += batch_size;
                total_time += gpu_timer.elapsed();
            }

            accuracy = 100.f * tp_count / sample_count;
            output_file << "[EVAL] avg loss: " << std::setw(4) << loss_val / sample_count
                        << ", accuracy: " << accuracy << "%"
                        << ", avg sample time: " << total_time / sample_count << "ms" << std::endl;
        }
    }
}

int main(int argc, char *argv[]) {
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    const int64_t batch_size = 128;
    const size_t num_epochs = 100;
    int monitoring_step = 20;
    const double learning_rate = 0.001;

    auto model = resnet50(NUMBER_CIFAR10_CLASSES);
    model->initialize_weights();
    model->to(device);

    std::stringstream ss;
    ss << "profiles/run_libtorch_" << argv[1] << "_" << argv[2] << ".csv";
    std::ofstream output_file(ss.str());

    if (strcmp(argv[1], "mnist") == 0) {
        std::cout << "== MNIST training with CUDNN ==" << std::endl;

        TestDataset<MNIST> train_dataset = MNIST("../data/MNIST/raw/", MNIST::Mode::kTrain)
                                               .map(torch::data::transforms::Stack<>());
        TestDataset<MNIST> test_dataset =
            MNIST("../data/MNIST/raw/", MNIST::Mode::kTest).map(torch::data::transforms::Stack<>());

        train<MNIST>(
            model,
            train_dataset,
            test_dataset,
            batch_size,
            monitoring_step,
            learning_rate,
            num_epochs,
            device,
            output_file);

    } else if (strcmp(argv[1], "cifar10") == 0) {
        std::cout << "== STL10 training with LibTorch ==" << std::endl;

        TestDataset<CIFAR10> train_dataset =
            CIFAR10("../data/cifar-10-batches-bin/all_train_data.bin")
                .map(torch::data::transforms::Stack<>());

        TestDataset<CIFAR10> test_dataset = CIFAR10("../data/cifar-10-batches-bin/test_batch.bin")
                                                .map(torch::data::transforms::Stack<>());

        train<CIFAR10>(
            model,
            train_dataset,
            test_dataset,
            batch_size,
            monitoring_step,
            learning_rate,
            num_epochs,
            device,
            output_file);
        //    } else if (strcmp(argv[1], "cifar10") == 0) {
        //        std::cout << "== CIFAR10 training with CUDNN ==" << std::endl;
        //        train_data_loader = new CIFAR10<float>(
        //            "../data/cifar-10-batches-bin/all_train_data.bin",
        //            "",
        //            true,
        //            batch_size,
        //            NUMBER_CIFAR10_CLASSES);
        //        test_data_loader = new CIFAR10<float>(
        //            "../data/cifar-10-batches-bin/test_batch.bin",
        //            "",
        //            false,
        //            batch_size,
        //            NUMBER_CIFAR10_CLASSES);
    } else {
        exit(EXIT_FAILURE);
    }
}
