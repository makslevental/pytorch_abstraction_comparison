//
// Created by Maksim Levental on 10/28/20.
//

#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <array>
#include <fstream>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <cudnn.h>

typedef enum { host, cuda } DeviceType;

template <typename ftype> class Tensor {
private:
    ftype *host_ptr_ = nullptr;
    ftype *device_ptr_ = nullptr;

    int batch_size_ = 1;
    int channels_ = 1;
    int height_ = 1;
    int width_ = 1;

    bool is_tensored_ = false;

    cudnnTensorDescriptor_t tensor_desc_ = nullptr;

public:
    ftype *get_host_ptr() const { return host_ptr_; }
    ftype *get_device_ptr() const { return device_ptr_; }
    int get_batch_size() const { return batch_size_; }
    int get_channels() const { return channels_; }
    int get_height() const { return height_; }
    int get_width() const { return width_; }
    const cudnnTensorStruct *get_tensor_desc() const { return tensor_desc_; }

    explicit Tensor(int n = 1, int c = 1, int h = 1, int w = 1)
        : batch_size_(n), channels_(c), height_(h), width_(w) {
        host_ptr_ = new float[batch_size_ * channels_ * height_ * width_];
    }
    explicit Tensor(std::array<int, 4> size)
        : batch_size_(size[0]), channels_(size[1]), height_(size[2]), width_(size[3]) {
        host_ptr_ = new float[batch_size_ * channels_ * height_ * width_];
    }

    ~Tensor() {
        if (host_ptr_ != nullptr)
            delete[] host_ptr_;
        if (device_ptr_ != nullptr)
            cudaFree(device_ptr_);
        if (is_tensored_)
            cudnnDestroyTensorDescriptor(tensor_desc_);
    }

    // reset the current blob with the new size information
    void reset(int n = 1, int c = 1, int h = 1, int w = 1) {
        // update size information
        batch_size_ = n;
        channels_ = c;
        height_ = h;
        width_ = w;

        // terminate current buffers
        if (host_ptr_ != nullptr) {
            delete[] host_ptr_;
            host_ptr_ = nullptr;
        }
        if (device_ptr_ != nullptr) {
            cudaFree(device_ptr_);
            device_ptr_ = nullptr;
        }

        // create new buffer
        host_ptr_ = new float[batch_size_ * channels_ * height_ * width_];
        cuda();

        // reset tensor descriptor if it was tensor
        if (is_tensored_) {
            cudnnDestroyTensorDescriptor(tensor_desc_);
            is_tensored_ = false;
        }
    }

    void reset(std::array<int, 4> size) { reset(size[0], size[1], size[2], size[3]); }

    // returns array of tensor shape
    std::array<int, 4> shape() {
        return std::array<int, 4>({batch_size_, channels_, height_, width_});
    }

    // returns number of elements for 1 batch
    int size() { return channels_ * height_ * width_; }

    // returns number of total elements in blob including batch
    int len() { return batch_size_ * channels_ * height_ * width_; }

    // returns size of allocated memory
    int buf_size() { return sizeof(ftype) * len(); }

    /* Tensor Control */
    cudnnTensorDescriptor_t tensor() {
        if (is_tensored_)
            return tensor_desc_;

        cudnnCreateTensorDescriptor(&tensor_desc_);
        cudnnSetTensor4dDescriptor(
            tensor_desc_,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            batch_size_,
            channels_,
            height_,
            width_);
        is_tensored_ = true;

        return tensor_desc_;
    }

    // get cuda memory
    ftype *cuda() {
        if (device_ptr_ == nullptr)
            cudaMalloc((void **)&device_ptr_, sizeof(ftype) * len());

        return device_ptr_;
    }

    // transfer data between memory
    ftype *to(DeviceType target) {
        if (target == host) {
            cudaMemcpy(host_ptr_, cuda(), sizeof(ftype) * len(), cudaMemcpyDeviceToHost);
            return host_ptr_;
        } else if (target == DeviceType::cuda) {
            cudaMemcpy(cuda(), host_ptr_, sizeof(ftype) * len(), cudaMemcpyHostToDevice);
            return device_ptr_;
        } else {
            return nullptr;
        }
    }

    void print(const std::string &name, bool view_param = false, int num_batch = 1) {
        to(host);
        std::cout << "**" << name << "\t: (" << size() << ")\t";
        std::cout << ".n: " << batch_size_ << ", .c: " << channels_ << ", .h: " << height_
                  << ", .w: " << width_;
        std::cout << std::hex << "\t(h:" << host_ptr_ << ", d:" << device_ptr_ << ")" << std::dec
                  << std::endl;

        if (view_param) {
            std::cout << std::fixed;
            std::cout.precision(6);

            int max_print_line = 4;
            if (width_ == 28) {
                std::cout.precision(3);
                max_print_line = 28;
            }
            int offset = 0;

            for (int n = 0; n < num_batch; n++) {
                if (num_batch > 1)
                    std::cout << "<--- batch[" << n << "] --->" << std::endl;
                int count = 0;
                int print_line_count = 0;
                while (count < size() && print_line_count < max_print_line) {
                    std::cout << "\t";
                    for (int s = 0; s < width_ && count < size(); s++) {
                        std::cout << host_ptr_[size() * n + count + offset] << "\t";
                        count++;
                    }
                    std::cout << std::endl;
                    print_line_count++;
                }
            }
            std::cout.unsetf(std::ios::fixed);
        }
    }

    /* pretrained parameter load and save */
    int file_read(const std::string &filename) {
        std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            std::cout << "fail to access " << filename << std::endl;
            return -1;
        }

        file.read((char *)host_ptr_, sizeof(float) * this->len());
        this->to(DeviceType::cuda);
        file.close();

        return 0;
    }

    int file_write(const std::string &filename) {
        std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            std::cout << "fail to write " << filename << std::endl;
            return -1;
        }
        file.write((char *)this->to(host), sizeof(float) * this->len());
        file.close();

        return 0;
    }
};

#endif // _TENSOR_H_
