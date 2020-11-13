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
#include <helper.h>

#include <prettyprint.h>

typedef enum { host, cuda } DeviceType;

template <typename dtype> class Tensor {
private:
    dtype *host_ptr_ = nullptr;
    dtype *device_ptr_ = nullptr;

    int batch_size_ = 1;
    int channels_ = 1;
    int height_ = 1;
    int width_ = 1;

    cudnnTensorDescriptor_t tensor_desc_ = nullptr;

public:
    dtype *get_host_ptr() const { return host_ptr_; }
    int get_batch_size() const { return batch_size_; }
    int get_channels() const { return channels_; }
    int get_height() const { return height_; }
    int get_width() const { return width_; }

    explicit Tensor(int n = 1, int c = 1, int h = 1, int w = 1)
        : batch_size_(n), channels_(c), height_(h), width_(w) {
        unsigned int n_elements = batch_size_ * channels_ * height_ * width_;
        const unsigned int bytes = n_elements * sizeof(float);
        checkCudaErrors(cudaMallocHost((void **)&host_ptr_, bytes));
        tensor_descriptor();
        checkCudaErrors(cudaMalloc((void **)&device_ptr_, sizeof(dtype) * len()));
    }
    explicit Tensor(std::array<int, 4> size)
        : batch_size_(size[0]), channels_(size[1]), height_(size[2]), width_(size[3]) {
        unsigned int n_elements = batch_size_ * channels_ * height_ * width_;
        const unsigned int bytes = n_elements * sizeof(float);
        checkCudaErrors(cudaMallocHost((void **)&host_ptr_, bytes));
        tensor_descriptor();
        checkCudaErrors(cudaMalloc((void **)&device_ptr_, sizeof(dtype) * len()));
    }

    Tensor(const Tensor<dtype> &t) {
        batch_size_ = t.batch_size_;
        channels_ = t.channels_;
        height_ = t.height_;
        width_ = t.width_;
        reset(shape());
        checkCudaErrors(cudaMalloc((void **)&device_ptr_, sizeof(dtype) * len()));
        download(t);
    }

    ~Tensor() {
        if (host_ptr_ != nullptr) {
            checkCudaErrors(cudaFreeHost(host_ptr_));
            host_ptr_ = nullptr;
        }
        if (device_ptr_ != nullptr) {
            checkCudaErrors(cudaFree(device_ptr_));
            device_ptr_ = nullptr;
        }
        if (tensor_desc_) {
            checkCudnnErrors(cudnnDestroyTensorDescriptor(tensor_desc_));
            tensor_desc_ = nullptr;
        }
    }

    void download(const Tensor<dtype> &t) {
        checkCudaErrors(cudaMemcpy(
            get_device_ptr(), t.get_device_ptr(), sizeof(dtype) * len(), cudaMemcpyDeviceToDevice));
        checkCudaErrors(
            cudaMemcpy(host_ptr_, t.get_host_ptr(), sizeof(dtype) * len(), cudaMemcpyHostToHost));
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
            checkCudaErrors(cudaFreeHost(host_ptr_));
            host_ptr_ = nullptr;
        }
        if (device_ptr_ != nullptr) {
            checkCudaErrors(cudaFree(device_ptr_));
            device_ptr_ = nullptr;
        }

        // create new buffer
        unsigned int n_elements = batch_size_ * channels_ * height_ * width_;
        const unsigned int bytes = n_elements * sizeof(float);
        checkCudaErrors(cudaMallocHost((void **)&host_ptr_, bytes));

        // reset tensor descriptor if it was tensor_descriptor
        if (tensor_desc_) {
            checkCudnnErrors(cudnnDestroyTensorDescriptor(tensor_desc_));
            tensor_desc_ = nullptr;
            tensor_descriptor();
        }
    }

    void reset(std::array<int, 4> size) { reset(size[0], size[1], size[2], size[3]); }
    void reset() { reset(batch_size_, channels_, height_, width_); }

    // returns array of tensor_descriptor shape
    std::array<int, 4> shape() {
        return std::array<int, 4>({batch_size_, channels_, height_, width_});
    }

    // returns number of elements for 1 batch
    int size() { return channels_ * height_ * width_; }

    // returns number of total elements in blob including batch
    int len() { return batch_size_ * channels_ * height_ * width_; }

    // returns size of allocated memory
    int buf_size() { return sizeof(dtype) * len(); }

    /* Tensor Control */
    cudnnTensorDescriptor_t tensor_descriptor() {
        if (tensor_desc_)
            return tensor_desc_;
        checkCudnnErrors(cudnnCreateTensorDescriptor(&tensor_desc_));
        checkCudnnErrors(cudnnSetTensor4dDescriptor(
            tensor_desc_,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            batch_size_,
            channels_,
            height_,
            width_));

        return tensor_desc_;
    }

    dtype *get_device_ptr() { return device_ptr_; }

    const dtype *get_device_ptr() const { return device_ptr_; }

    // transfer data between memory
    dtype *to(DeviceType target) {
        // TODO: keep track of where the canonical data is using a flag/enum
        if (target == host) {
            checkCudaErrors(cudaMemcpy(
                host_ptr_, get_device_ptr(), sizeof(dtype) * len(), cudaMemcpyDeviceToHost));
            return host_ptr_;
        } else if (target == DeviceType::cuda) {
            checkCudaErrors(cudaMemcpy(
                get_device_ptr(), host_ptr_, sizeof(dtype) * len(), cudaMemcpyHostToDevice));
            return device_ptr_;
        } else {
            exit(EXIT_FAILURE);
        }
    }

    void print(const std::string &name, bool view_param = false, int num_batch = 1) {
        // TODO: copy to host without overwriting
        std::cout << "**" << name << "\t: (" << size() << ")\t";
        std::cout << ".n: " << batch_size_ << ", .c: " << channels_ << ", .h: " << height_
                  << ", .w: " << width_;
        std::cout << std::hex << "\t(h:" << host_ptr_ << ", d:" << device_ptr_ << ")" << std::dec
                  << std::endl;

        if (view_param) {
            std::cout << std::fixed;
            std::cout.precision(6);

            int max_print_line = 100;
            if (width_ == 28) {
                std::cout.precision(3);
                max_print_line = 28;
            }
            int offset = 0;

            for (int n = 0; n < num_batch; n++) {
                std::cout << "<--- batch[" << n << "] --->" << std::endl;
                int count = 0;
                int print_line_count = 0;
                while (count < size() && print_line_count < max_print_line) {
                    std::cout << "\t";
                    for (int s = 0; s < width_ && count < size(); s++) {
                        if (width_ == 28) {
                            if (host_ptr_[size() * n + count + offset] > 0)
                                std::cout << "*";
                            else
                                std::cout << " ";
                        } else {
                            std::cout << host_ptr_[size() * n + count + offset] << "\t";
                        }
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
