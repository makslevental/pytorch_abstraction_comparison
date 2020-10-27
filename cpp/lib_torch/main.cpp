#include "resnet.h"
#include "util.h"
#include <iostream>
#include <torch/torch.h>

using namespace torch;

// int main() {
//     auto const kNoiseSize = 100;
//     auto const kBatchSize = 32;
//     auto const kNumberOfEpochs = 10;
//     auto const kCheckpointEvery = 1000;

//     torch::Device device = torch::kCPU;
//     if (torch::cuda::is_available()) {
//         std::cout << "CUDA is available! Training on GPU." << std::endl;
//         device = torch::kCUDA;
//     }

//     DCGANGenerator generator(kNoiseSize);
//     auto discriminator = buildDiscriminator();
//     generator->to(device);
//     discriminator->to(device);

//     auto dataset = torch::data::datasets::MNIST("/home/maksim/dev_projects/"
//                                                 "pytorch_abstraction_comparison/data/mnist")
//                        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
//                        .map(torch::data::transforms::Stack<>());
//     auto const batches_per_epoch = dataset.size().value_or(0) / kNumberOfEpochs;

//     auto data_loader = torch::data::make_data_loader(
//         std::move(dataset), torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(4));

//     torch::optim::Adam generator_optimizer(
//         generator->parameters(),
//         torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.999)));
//     torch::optim::Adam discriminator_optimizer(
//         discriminator->parameters(),
//         torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.999)));

//     auto checkpoint_counter = 0;
//     for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
//         int64_t batch_index = 0;
//         for (torch::data::Example<> &batch : *data_loader) {
//             // Train discriminator with real images.
//             discriminator->zero_grad();

//             Tensor real_images = batch.data.to(device);
//             Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
//             Tensor real_output = discriminator->forward(real_images);
//             Tensor d_loss_real = binary_cross_entropy(real_output, real_labels);
//             d_loss_real.backward();

//             // Train discriminator with fake images.
//             Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
//             Tensor fake_images = generator->forward(noise);
//             Tensor fake_labels = torch::zeros(batch.data.size(0), device);
//             Tensor fake_output = discriminator->forward(fake_images.detach());
//             Tensor d_loss_fake = binary_cross_entropy(fake_output, fake_labels);
//             d_loss_fake.backward();

//             Tensor d_loss = d_loss_real + d_loss_fake;
//             discriminator_optimizer.step();

//             // Train generator.
//             generator->zero_grad();
//             fake_labels.fill_(1);
//             fake_output = discriminator->forward(fake_images);
//             Tensor g_loss = binary_cross_entropy(fake_output, fake_labels);
//             g_loss.backward();
//             generator_optimizer.step();

//             std::printf("\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f", epoch,
//                         kNumberOfEpochs, ++batch_index, batches_per_epoch, d_loss.item<float>(),
//                         g_loss.item<float>());

//             if (batch_index % kCheckpointEvery == 0) {
//                 // Checkpoint the model and optimizer state.
//                 torch::save(generator, "generator-checkpoint.pt");
//                 torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
//                 torch::save(discriminator, "discriminator-checkpoint.pt");
//                 torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
//                 // Sample the generator and save the images.
//                 torch::Tensor samples =
//                     generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
//                 torch::save((samples + 1.0) / 2.0,
//                             torch::str("dcgan-sample-", checkpoint_counter++, ".pt"));
//                 std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
//             }
//         }
//     }
// }

int main() {
    torch::Device device("cpu");
    if (torch::cuda::is_available()) {
        device = torch::Device("cuda:0");
    }
    std::cout << device << std::endl;

    torch::Tensor t = torch::rand({2, 3, 224, 224}).to(device);

    auto resnet = resnet50(10);
    // print_modules(resnet.ptr());
    resnet->initialize_weights();
    resnet->to(device);
    t = resnet->forward(t);
    std::cout << t.sizes() << std::endl;
}


