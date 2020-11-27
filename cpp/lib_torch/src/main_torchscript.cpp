#include "cifar10.h"
#include "cuda_helper.h"
#include "cuda_profiling.h"
#include "nvml.h"
#include "pascal.h"
#include "transform.h"

#include <cuda_profiler_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <torch/script.h>
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
    torch::jit::Module *model,
    TestDataset<DatasetType> train_dataset,
    TestDataset<DatasetType> test_dataset,
    int64_t batch_size,
    int monitoring_step,
    double learning_rate,
    size_t num_epochs,
    torch::Device device,
    std::ostream &output_file) {

    auto num_test_samples = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    std::string nvtx_message;
    auto gpu_timer = GPUTimer();
    cudaProfilerStart();

    double loss_val, accuracy, running_loss;
    int tp_count, running_tp_count, running_sample_count, sample_count;
    double total_time;
    double elapsed_time;
    double used_mem = 0, running_used_mem = 0;
    int batch_n = 0;
    std::vector<torch::jit::IValue> inputs;

    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        model->eval();
        torch::NoGradGuard no_grad;

        total_time = sample_count = tp_count = loss_val = 0;
        used_mem = 0;

        for (const auto &batch : *test_loader) {
            nvtx_message = std::string(
                "eval epoch " + std::to_string(epoch) + " batch " +
                std::to_string(batch_n));
            nvtxRangePushA(nvtx_message.c_str());
            nvtxRangePushA("batch load");

            gpu_timer.start();

            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            nvtxRangePop();

            inputs.clear();
            inputs.push_back(data);
            auto output = model->forward(inputs).toTensor();
            auto loss = torch::nn::functional::cross_entropy(output, target);
            auto prediction = output.argmax(1);
            tp_count += prediction.eq(target).sum().template item<int64_t>();

            nvtxRangePop();

            gpu_timer.stop();

            loss_val += loss.template item<double>();
            sample_count += batch_size;
            total_time += gpu_timer.elapsed();
            used_mem += get_used_cuda_mem();
        }

        output_file << "[EVAL] avg loss: " << std::setw(4) << loss_val / sample_count
                    << ", accuracy: " << 100.f * tp_count / sample_count << "%"
                    << ", avg sample time: " << total_time / sample_count << "ms"
                    << std::defaultfloat
                    << ", avg used mem: " << used_mem / (sample_count / batch_size) << "mb"
                    << ", avg gpu util: " << get_gpu_utilization() << "%" << std::endl;
        output_file.flush();

    }
}

int main(int argc, char *argv[]) {
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    int64_t batch_size = std::stoi(std::getenv("BATCH_SIZE"));
    int num_classes;
    int num_epochs = std::stoi(std::getenv("EPOCHS"));
    int monitoring_step = 20;

    double learning_rate = 1.0 / std::stoi(std::getenv("INV_LEARNING_RATE"));

    torch::jit::script::Module torch_script_model;
    try {
        std::stringstream ss;
        ss << "traced_resnet50_" << std::getenv("RESOLUTION") << ".pt";
        // Deserialize the ScriptModule from a file using torch::jit::load().
        torch_script_model = torch::jit::load(ss.str());
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        exit(EXIT_FAILURE);
    }

    std::stringstream ss;
    ss << "profiles/run_torchscript"
       << "_" << batch_size << "_" << std::getenv("RESOLUTION") << ".csv";
    std::ofstream output_file(ss.str());

    if (strcmp(argv[1], "pascal") == 0) {
        auto model = &torch_script_model;
        model->to(device);

        std::cout << "== PASCAL training with TorchScript ==" << std::endl;
        TestDataset<PASCAL> train_dataset = PASCAL("../data/VOCdevkit/VOC2012", PASCAL::kTrain)
                                                .map(torch::data::transforms::Stack<>());
        TestDataset<PASCAL> test_dataset = PASCAL("../data/VOCdevkit/VOC2012", PASCAL::kVal)
                                               .map(torch::data::transforms::Stack<>());

        train<PASCAL>(
            model,
            train_dataset,
            test_dataset,
            batch_size,
            monitoring_step,
            learning_rate,
            num_epochs,
            device,
            output_file);
    } else {
        exit(EXIT_FAILURE);
    }
}
