#include <cassert>
#include <mnist.h>
// TODO: multithreading to match pytorch?
void MNIST::load_data() {
    uint8_t ptr[4];

    std::cout << "loading " << dataset_fp_ << std::endl;
    std::ifstream file(dataset_fp_.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Download dataset first!!" << std::endl;
        std::cout << "You can get the MNIST dataset from 'http://yann.lecun.com/exdb/mnist/' or "
                     "just use 'download_mnist.sh' file."
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    file.read((char *)ptr, 4);
    int magic_number = to_int(ptr);
    assert((magic_number & 0xFFF) == 0x803);

    int num_data;
    file.read((char *)ptr, 4);
    num_data = to_int(ptr);
    file.read((char *)ptr, 4);
    height_ = to_int(ptr);
    file.read((char *)ptr, 4);
    width_ = to_int(ptr);
    channels_ = 1;

    auto *q = new uint8_t[channels_ * height_ * width_];
    for (int i = 0; i < num_data; i++) {
        std::vector<float> image = std::vector<float>(channels_ * height_ * width_);
        float *image_ptr = image.data();

        file.read((char *)q, channels_ * height_ * width_);
        for (int j = 0; j < channels_ * height_ * width_; j++) {
            image_ptr[j] = (float)q[j];
        }

        data_pool_.push_back(image);
    }

    delete q;

    num_batches_ = num_data / batch_size_;
    std::cout << "num_batches: " << num_batches_ << std::endl;
    std::cout << "loaded " << data_pool_.size() << " items.." << std::endl;

    file.close();
}

void MNIST::load_target() {
    uint8_t ptr[4];

    std::ifstream file(label_fp_.c_str(), std::ios::in | std::ios::binary);
    std::cout << "loading " << label_fp_ << std::endl;

    if (!file.is_open()) {
        std::cout << "Check dataset existance!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    file.read((char *)ptr, 4);
    int magic_number = to_int(ptr);
    assert((magic_number & 0xFFF) == 0x801);

    file.read((char *)ptr, 4);
    int num_target = to_int(ptr);

    // prepare input buffer for label
    // read all labels and converts to one-hot encoding
    for (int i = 0; i < num_target; i++) {
        std::vector<float> target_one_hot(num_classes_, 0.f);
        file.read((char *)ptr, 1);
        target_one_hot[static_cast<int>(ptr[0])] = 1.f;
        target_pool_.push_back(target_one_hot);
    }

    file.close();
}

void MNIST::normalize_data() {
    for (auto &sample : data_pool_) {
        float *sample_data_ptr = sample.data();
        for (int j = 0; j < channels_ * height_ * width_; j++) {
            sample_data_ptr[j] /= 255.f;
        }
    }
}

MNIST::MNIST(
    const string &dataset_fp, const string &label_fp, bool shuffle, int batch_size, int num_classes)
    : Dataset(dataset_fp, label_fp, shuffle, batch_size, num_classes) {

    // https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP50-CPP.+Do+not+invoke+virtual+functions+from+constructors+or+destructors
    MNIST::load_data();
    MNIST::normalize_data();
    MNIST::load_target();

    if (shuffle_)
        shuffle_dataset();
    create_shared_space();
}

int MNIST::get_num_batches() const { return num_batches_; }
