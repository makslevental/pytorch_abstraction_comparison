//
// Created by Maksim Levental on 10/29/20.
//

#include <conv_2d.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <vector>

/**
 * Convolutional layer with bias
 */
Conv2D::Conv2D(
    std::string name,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool bias)
    : out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding),
      dilation_(dilation), bias_(bias) {
    name_ = std::move(name);

    // create cudnn container handles
    cudnnCreateFilterDescriptor(&filter_desc_);

    cudnnCreateConvolutionDescriptor(&conv_desc_);
    checkCudnnErrors(cudnnSetConvolution2dDescriptor(
        conv_desc_,
        padding_,
        padding_,
        stride_,
        stride_,
        dilation_,
        dilation_,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    // setting cudnn convolution math type
    // CUDNN_DEFAULT_MATH operates convolution with FP32.
    // If you use A100, CUDNN utilise tensor_descriptor cores with TF32.
    checkCudnnErrors(cudnnSetConvolutionMathType(conv_desc_, CUDNN_DEFAULT_MATH));

    device_workspace_ = nullptr;
}

Conv2D::~Conv2D() {
    // distroy cudnn container resources
    cudnnDestroyFilterDescriptor(filter_desc_);
    cudnnDestroyConvolutionDescriptor(conv_desc_);

    // terminate internal created blobs
    if (device_workspace_ != nullptr) {
        cudaFree(device_workspace_);
        device_workspace_ = nullptr;
    }
}

void Conv2D::set_workspace() {
    size_t temp_size = 0;

    // forward
#if CUDNN_MAJOR >= 7
    std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_algo_perf_results(
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algo_perf_results(
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algo_perf_results(
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);

    int algo_max_count;
    int returnedAlgoCount = 0;
    checkCudnnErrors(cudnnGetConvolutionForwardAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
    std::cout << this->name_ << ": Available Algorithm Count [FWD]: " << algo_max_count
              << std::endl;
    checkCudnnErrors(cudnnFindConvolutionForwardAlgorithm(
        cuda_->cudnn(),
        input_desc_,
        filter_desc_,
        conv_desc_,
        output_desc_,
        algo_max_count,
        &returnedAlgoCount,
        &fwd_algo_perf_results[0]));
    std::cout << "returned algo_count: " << returnedAlgoCount << std::endl;
    for (int i = 0; i < returnedAlgoCount; i++)
        std::cout << "fwd algo[" << i << "] time: " << fwd_algo_perf_results[i].time
                  << ", memory: " << fwd_algo_perf_results[i].memory << std::endl;
#else
    checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm_v7(
        cuda_->cudnn(),
        input_desc_,
        filter_desc_,
        conv_desc_,
        output_desc_,
        algo_max_count,
        &returnedAlgoCount,
        &fwd_algo_perf_results[0]));
#endif
    // choose the fastest algorithm
    conv_fwd_algo_ = fwd_algo_perf_results[0].algo;
#else
    checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm(
        cuda_->cudnn(),
        input_desc_,
        filter_desc_,
        conv_desc_,
        output_desc_,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &conv_fwd_algo_));
#endif
    checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(
        cuda_->cudnn(),
        input_desc_,
        filter_desc_,
        conv_desc_,
        output_desc_,
        conv_fwd_algo_,
        &temp_size));
    workspace_size_ = std::max(workspace_size_, temp_size);

    // bwd - filter
#if CUDNN_MAJOR >= 7
    checkCudnnErrors(
        cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
    std::cout << this->name_ << ": Available Algorithm Count [BWD-filter]: " << algo_max_count
              << std::endl;
    checkCudnnErrors(cudnnFindConvolutionBackwardFilterAlgorithm(
        cuda_->cudnn(),
        input_desc_,
        output_desc_,
        conv_desc_,
        filter_desc_,
        algo_max_count,
        &returnedAlgoCount,
        &bwd_filter_algo_perf_results[0]));
    for (int i = 0; i < returnedAlgoCount; i++)
        std::cout << "bwd filter algo[" << i << "] time: " << fwd_algo_perf_results[i].time
                  << ", memory: " << fwd_algo_perf_results[i].memory << std::endl;
#else
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        cuda_->cudnn(),
        input_desc_,
        output_desc_,
        conv_desc_,
        filter_desc_,
        algo_max_count,
        &returnedAlgoCount,
        &bwd_filter_algo_perf_results[0]));
#endif
    conv_bwd_filter_algo_ = bwd_filter_algo_perf_results[0].algo;
#else
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm(
        cuda_->cudnn(),
        input_desc_,
        output_desc_,
        conv_desc_,
        filter_desc_,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        &conv_bwd_filter_algo_));
#endif
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cuda_->cudnn(),
        input_desc_,
        output_desc_,
        conv_desc_,
        filter_desc_,
        conv_bwd_filter_algo_,
        &temp_size));
    workspace_size_ = std::max(workspace_size_, temp_size);

    // bwd - data
#if CUDNN_MAJOR >= 7
    checkCudnnErrors(
        cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
    std::cout << this->name_ << ": Available Algorithm Count [BWD-data]: " << algo_max_count
              << std::endl;
    checkCudnnErrors(cudnnFindConvolutionBackwardDataAlgorithm(
        cuda_->cudnn(),
        filter_desc_,
        output_desc_,
        conv_desc_,
        input_desc_,
        algo_max_count,
        &returnedAlgoCount,
        &bwd_data_algo_perf_results[0]));
    for (int i = 0; i < returnedAlgoCount; i++)
        std::cout << "bwd data algo[" << i << "] time: " << fwd_algo_perf_results[i].time
                  << ", memory: " << fwd_algo_perf_results[i].memory << std::endl;
#else
    checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        cuda_->cudnn(),
        filter_desc_,
        output_desc_,
        conv_desc_,
        input_desc_,
        algo_max_count,
        &returnedAlgoCount,
        &bwd_data_algo_perf_results[0]));
#endif
    conv_bwd_data_algo_ = bwd_data_algo_perf_results[0].algo;
#else
    checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm(
        cuda_->cudnn(),
        filter_desc_,
        output_desc_,
        conv_desc_,
        input_desc_,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &conv_bwd_data_algo_));
#endif
    checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cuda_->cudnn(),
        filter_desc_,
        output_desc_,
        conv_desc_,
        input_desc_,
        conv_bwd_data_algo_,
        &temp_size));
    workspace_size_ = std::max(workspace_size_, temp_size);

    if (workspace_size_ > 0) {
        if (device_workspace_ != nullptr)
            checkCudaErrors(cudaFree(device_workspace_));
        checkCudaErrors(cudaMalloc((void **)&device_workspace_, workspace_size_));
    }
}

void Conv2D::fwd_initialize(Tensor<float> *input) {
    // initialize weights and bias
    if (weights_ == nullptr) {
        // initialize containers handles
        checkCudnnErrors(cudnnSetFilter4dDescriptor(
            filter_desc_,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            out_channels_,
            input->get_channels(),
            kernel_size_,
            kernel_size_));

        weights_ =
            new Tensor<float>(out_channels_, input->get_channels(), kernel_size_, kernel_size_);
        if (bias_) {
            biases_ = new Tensor<float>(1, out_channels_); // bias size
            bias_desc_ = biases_->tensor_descriptor();
        }
    }

    // initilaize input and output
    if (input_ == nullptr || batch_size_ != input->get_batch_size()) {
        // initialize input
        input_ = input;
        input_desc_ = input->tensor_descriptor();
        batch_size_ = input->get_batch_size();

        // initilaize output
        checkCudnnErrors(cudnnGetConvolution2dForwardOutputDim(
            conv_desc_,
            input_desc_,
            filter_desc_,
            &output_size_[0],
            &output_size_[1],
            &output_size_[2],
            &output_size_[3]));

        if (output_ == nullptr)
            output_ = new Tensor<float>(output_size_);
        else
            output_->reset(output_size_);

        output_desc_ = output_->tensor_descriptor();

        // initialize workspace for cudnn
        set_workspace();

        // initialize weights
        if (load_pretrain_ && !freeze_) {
            if (load_parameter()) {
                std::cout << "error occurred.." << std::endl;
                exit(-1);
            }
        } else if (!freeze_) {
            init_weight_bias();
        } else {
            /* do nothing */
        }
    }
}

Tensor<float> *Conv2D::forward(Tensor<float> *input) {
    checkCudnnErrors(cudnnConvolutionForward(
        cuda_->cudnn(),
        &cuda_->one,
        input_desc_,
        input_->get_device_ptr(),
        filter_desc_,
        weights_->get_device_ptr(),
        conv_desc_,
        conv_fwd_algo_,
        device_workspace_,
        workspace_size_,
        &cuda_->zero,
        output_desc_,
        output_->get_device_ptr()));
    if (bias_) {
        checkCudnnErrors(cudnnAddTensor(
            cuda_->cudnn(),
            &cuda_->one,
            bias_desc_,
            biases_->get_device_ptr(),
            &cuda_->one,
            output_desc_,
            output_->get_device_ptr()));
    }

#if (DEBUG_CONV & 0x01)
    input_->print(name_ + "::input", true, input_->get_batch_size());
    weights_->print(name_ + "::weight", true);
    biases_->print(name_ + "::bias", true);
    output_->print(name_ + "::output", true);
#endif

    return output_;
}

void Conv2D::bwd_initialize(Tensor<float> *grad_output) {
    if (grad_weights_ == nullptr) {
        grad_weights_ = new Tensor<float>(weights_->shape());
        if (bias_) {
            grad_biases_ = new Tensor<float>(1, biases_->get_channels());
        }
    }
    Layer::bwd_initialize(grad_output);
}

Tensor<float> *Conv2D::backward(Tensor<float> *grad_output) {
    // gradients of biases
    if (bias_) {
        checkCudnnErrors(cudnnConvolutionBackwardBias(
            cuda_->cudnn(),
            &cuda_->one,
            output_desc_,
            grad_output->get_device_ptr(),
            &cuda_->zero,
            bias_desc_,
            grad_biases_->get_device_ptr()));
    }

    // gradients of weights
    checkCudnnErrors(cudnnConvolutionBackwardFilter(
        cuda_->cudnn(),
        &cuda_->one,
        input_desc_,
        input_->get_device_ptr(),
        output_desc_,
        grad_output_->get_device_ptr(),
        conv_desc_,
        conv_bwd_filter_algo_,
        device_workspace_,
        workspace_size_,
        &cuda_->zero,
        filter_desc_,
        grad_weights_->get_device_ptr()));

    // gradients of input data
    if (!gradient_stop_)
        checkCudnnErrors(cudnnConvolutionBackwardData(
            cuda_->cudnn(),
            &cuda_->one,
            filter_desc_,
            weights_->get_device_ptr(),
            output_desc_,
            grad_output->get_device_ptr(),
            conv_desc_,
            conv_bwd_data_algo_,
            device_workspace_,
            workspace_size_,
            &cuda_->zero,
            input_desc_,
            grad_input_->get_device_ptr()));

#if (DEBUG_CONV & 0x02)
    std::cout << name_ << "[BACKWARD]" << std::endl;
    grad_output->print(name_ + "::gradients", true);
    grad_biases_->print(name_ + "gbias", true);
    grad_weights_->print(name_ + "gfilter", true);
    if (!gradient_stop_)
        grad_input_->print(name_ + "gdata", true);
#endif

#if (DEBUG_CONV & 0x04)
    grad_output->print(name_ + "::gradients", true);
    grad_biases_->print(name_ + "::gbias", true);
#endif

    return grad_input_;
}
