#include <cassert>
#include <datasets/mnist.h>
// TODO: multithreading to match pytorch?

template <typename dtype> void MNIST<dtype>::load_data() {
    uint8_t ptr[4];

    std::cout << "loading " << this->dataset_fp_ << std::endl;
    std::ifstream file(this->dataset_fp_.c_str(), std::ios::in | std::ios::binary);
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
    this->height_ = to_int(ptr);
    file.read((char *)ptr, 4);
    this->width_ = to_int(ptr);
    this->channels_ = 1;

    auto *q = new uint8_t[this->channels_ * this->height_ * this->width_];
    for (int i = 0; i < num_data; i++) {
        std::vector<dtype> image =
            std::vector<dtype>(this->channels_ * this->height_ * this->width_);
        dtype *image_ptr = image.data();

        file.read((char *)q, this->channels_ * this->height_ * this->width_);
        for (int j = 0; j < this->channels_ * this->height_ * this->width_; j++) {
            image_ptr[j] = (dtype)q[j];
        }

        this->data_pool_.push_back(image);
    }

    delete q;

    this->num_batches_ = num_data / this->batch_size_;
    std::cout << "num_batches: " << this->num_batches_ << std::endl;
    std::cout << "loaded " << this->data_pool_.size() << " items.." << std::endl;

    file.close();
}

template <typename dtype> void MNIST<dtype>::load_target() {
    uint8_t ptr[4];

    std::ifstream file(this->label_fp_.c_str(), std::ios::in | std::ios::binary);
    std::cout << "loading " << this->label_fp_ << std::endl;

    if (!file.is_open()) {
        std::cout << "Check dataset existance!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    file.read((char *)ptr, 4);
    int magic_number = to_int(ptr);
    assert((magic_number & 0xFFF) == 0x801);

    file.read((char *)ptr, 4);
    int num_target = to_int(ptr);

    // read all labels and converts to one-hot encoding
    for (int i = 0; i < num_target; i++) {
        std::vector<dtype> target_one_hot(this->num_classes_, 0.f);
        file.read((char *)ptr, 1);
        target_one_hot[static_cast<int>(ptr[0])] = 1.f;
        this->target_pool_.push_back(target_one_hot);
    }

    file.close();
}

template <typename dtype> void MNIST<dtype>::normalize_data() {
    for (auto &sample : this->data_pool_) {
        dtype *sample_data_ptr = sample.data();
        for (int j = 0; j < this->channels_ * this->height_ * this->width_; j++) {
            sample_data_ptr[j] /= 255.f;
        }
    }
}

template <typename dtype>
MNIST<dtype>::MNIST(
    const string &dataset_fp, const string &label_fp, bool shuffle, int batch_size, int num_classes)
    : Dataset<dtype>(dataset_fp, label_fp, shuffle, batch_size, num_classes) {

    // https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP50-CPP.+Do+not+invoke+virtual+functions+from+constructors+or+destructors
    MNIST::load_data();
    MNIST::normalize_data();
    MNIST::load_target();

    if (this->shuffle_)
        this->shuffle_dataset();
    this->create_shared_space();
}

template <typename dtype> int MNIST<dtype>::get_num_batches() const { return this->num_batches_; }

template class MNIST<float>;
template class MNIST<double>;
