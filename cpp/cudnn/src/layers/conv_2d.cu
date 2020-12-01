//
// Created by Maksim Levental on 10/29/20.
//

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <layers/conv_2d.cuh>
#include <vector>

/**
 * Convolutional layer with bias
 */
template <typename dtype>
Conv2d<dtype>::Conv2d(
    std::string name,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool bias)
    : out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding),
      dilation_(dilation), bias_(bias) {

    this->name_ = std::move(name);

    // create cudnn container handles
    cudnnCreateFilterDescriptor(&this->filter_desc_);

    cudnnCreateConvolutionDescriptor(&conv_desc_);
    cudnnDataType_t t;
    if constexpr (std::is_same<dtype, float>{}) {
        t = CUDNN_DATA_FLOAT;
    } else if constexpr (std::is_same<dtype, double>{}) {
        t = CUDNN_DATA_DOUBLE;
    }
    checkCudnnErrors(cudnnSetConvolution2dDescriptor(
        conv_desc_,
        padding_,
        padding_,
        stride_,
        stride_,
        dilation_,
        dilation_,
        CUDNN_CROSS_CORRELATION,
        t));

    // setting cudnn convolution math type
    // CUDNN_DEFAULT_MATH operates convolution with FP32.
    // If you use A100, CUDNN utilise tensor_descriptor cores with TF32.
    checkCudnnErrors(cudnnSetConvolutionMathType(conv_desc_, CUDNN_DEFAULT_MATH));

    device_workspace_ = nullptr;
}

template <typename dtype> Conv2d<dtype>::~Conv2d() {
    // distroy cudnn container resources
    cudnnDestroyFilterDescriptor(this->filter_desc_);
    cudnnDestroyConvolutionDescriptor(conv_desc_);

    // terminate internal created blobs
    if (device_workspace_ != nullptr) {
        cudaFree(device_workspace_);
        device_workspace_ = nullptr;
    }
}

template <typename dtype> void Conv2d<dtype>::set_workspace() {
    size_t temp_size = 0;

    // forward
    std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_algo_perf_results(
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algo_perf_results(
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algo_perf_results(
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);

    int algo_max_count;
    int returnedAlgoCount = 0;
    checkCudnnErrors(
        cudnnGetConvolutionForwardAlgorithmMaxCount(this->cuda_->cudnn(), &algo_max_count));
    checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm_v7(
        this->cuda_->cudnn(),
        this->input_desc_,
        this->filter_desc_,
        conv_desc_,
        this->output_desc_,
        algo_max_count,
        &returnedAlgoCount,
        &fwd_algo_perf_results[0]));
    // choose the fastest algorithm
    conv_fwd_algo_ = fwd_algo_perf_results[0].algo;
    checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(
        this->cuda_->cudnn(),
        this->input_desc_,
        this->filter_desc_,
        conv_desc_,
        this->output_desc_,
        conv_fwd_algo_,
        &temp_size));
    workspace_size_ = std::max(workspace_size_, temp_size);

    // bwd - filter
    checkCudnnErrors(
        cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(this->cuda_->cudnn(), &algo_max_count));
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        this->cuda_->cudnn(),
        this->input_desc_,
        this->output_desc_,
        conv_desc_,
        this->filter_desc_,
        algo_max_count,
        &returnedAlgoCount,
        &bwd_filter_algo_perf_results[0]));
    conv_bwd_filter_algo_ = bwd_filter_algo_perf_results[0].algo;
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        this->cuda_->cudnn(),
        this->input_desc_,
        this->output_desc_,
        conv_desc_,
        this->filter_desc_,
        conv_bwd_filter_algo_,
        &temp_size));
    workspace_size_ = std::max(workspace_size_, temp_size);

    // bwd - data
    checkCudnnErrors(
        cudnnGetConvolutionBackwardDataAlgorithmMaxCount(this->cuda_->cudnn(), &algo_max_count));
    checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        this->cuda_->cudnn(),
        this->filter_desc_,
        this->output_desc_,
        conv_desc_,
        this->input_desc_,
        algo_max_count,
        &returnedAlgoCount,
        &bwd_data_algo_perf_results[0]));
    conv_bwd_data_algo_ = bwd_data_algo_perf_results[0].algo;
    checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(
        this->cuda_->cudnn(),
        this->filter_desc_,
        this->output_desc_,
        conv_desc_,
        this->input_desc_,
        conv_bwd_data_algo_,
        &temp_size));
    workspace_size_ = std::max(workspace_size_, temp_size);

    if (workspace_size_ > 0) {
        if (device_workspace_ != nullptr)
            checkCudaErrors(cudaFree(device_workspace_));
        checkCudaErrors(cudaMalloc((void **)&device_workspace_, workspace_size_));
    }
}

template <typename dtype> void Conv2d<dtype>::fwd_initialize(Tensor<dtype> *input) {
    // initialize weights and bias
    if (this->weights_ == nullptr) {
        // initialize containers handles
        cudnnDataType_t t;
        if constexpr (std::is_same<dtype, float>{}) {
            t = CUDNN_DATA_FLOAT;
        } else if constexpr (std::is_same<dtype, double>{}) {
            t = CUDNN_DATA_DOUBLE;
        }

        checkCudnnErrors(cudnnSetFilter4dDescriptor(
            this->filter_desc_,
            t,
            CUDNN_TENSOR_NCHW,
            out_channels_,
            input->get_channels(),
            kernel_size_,
            kernel_size_));

        this->weights_ =
            new Tensor<dtype>(out_channels_, input->get_channels(), kernel_size_, kernel_size_);
        if (bias_) {
            this->biases_ = new Tensor<dtype>(1, out_channels_); // bias size
            this->bias_desc_ = this->biases_->tensor_descriptor();
        }
    }

    // initilaize input and output
    if (this->input_desc_ == nullptr || this->batch_size_ != input->get_batch_size()) {
        // initialize input
        this->input_size_ = input->size();
        this->input_desc_ = input->tensor_descriptor();
        this->batch_size_ = input->get_batch_size();
        this->in_channels_ = input->get_channels();

        // initilaize output
        checkCudnnErrors(cudnnGetConvolution2dForwardOutputDim(
            conv_desc_,
            this->input_desc_,
            this->filter_desc_,
            &output_size_[0],
            &output_size_[1],
            &output_size_[2],
            &output_size_[3]));

        if (this->output_ == nullptr)
            this->output_ = new Tensor<dtype>(output_size_);
        else
            this->output_->reset(output_size_);

        this->output_desc_ = this->output_->tensor_descriptor();

        // initialize workspace for cudnn
        set_workspace();

        // initialize weights
        if (this->load_pretrain_ && !this->freeze_) {
            if (this->load_parameter()) {
                std::cout << "error occurred.." << std::endl;
                exit(EXIT_FAILURE);
            }
        } else if (!this->freeze_) {
            this->init_weight_bias();
//            this->weights_->print("weights", true);
//            this->biases_->print("biases", true);
        } else {
            /* do nothing */
        }
    }
}

template <typename dtype> Tensor<dtype> *Conv2d<dtype>::forward(Tensor<dtype> *input) {
    fwd_initialize(input);
    this->input_ = input;
    checkCudnnErrors(cudnnConvolutionForward(
        this->cuda_->cudnn(),
        &this->cuda_->one,
        this->input_desc_,
        input->get_device_ptr(),
        this->filter_desc_,
        this->weights_->get_device_ptr(),
        conv_desc_,
        conv_fwd_algo_,
        device_workspace_,
        workspace_size_,
        &this->cuda_->zero,
        this->output_desc_,
        this->output_->get_device_ptr()));
    if (bias_) {
        checkCudnnErrors(cudnnAddTensor(
            this->cuda_->cudnn(),
            &this->cuda_->one,
            this->bias_desc_,
            this->biases_->get_device_ptr(),
            &this->cuda_->one,
            this->output_desc_,
            this->output_->get_device_ptr()));
    }

    if (DEBUG_CONV & 0x01) {
        input->print(this->name_ + "::input", true, input->get_batch_size());
        this->weights_->print(this->name_ + "::weight", true);
        this->biases_->print(this->name_ + "::bias", true);
        this->output_->print(this->name_ + "::output", true);
    }

    return this->output_;
}

template <typename dtype> void Conv2d<dtype>::bwd_initialize(Tensor<dtype> *grad_output) {
    if (this->grad_weights_ == nullptr) {
        this->grad_weights_ = new Tensor<dtype>(this->weights_->shape());
        if (bias_) {
            this->grad_biases_ = new Tensor<dtype>(1, this->biases_->get_channels());
        }
    }
    Layer<dtype>::bwd_initialize(grad_output);
}

template <typename dtype> Tensor<dtype> *Conv2d<dtype>::backward(Tensor<dtype> *grad_output) {
    bwd_initialize(grad_output);
    // gradients of biases
    if (bias_) {
        checkCudnnErrors(cudnnConvolutionBackwardBias(
            this->cuda_->cudnn(),
            &this->cuda_->one,
            this->output_desc_,
            grad_output->get_device_ptr(),
            &this->cuda_->zero,
            this->bias_desc_,
            this->grad_biases_->get_device_ptr()));
    }

    // gradients of weights
    checkCudnnErrors(cudnnConvolutionBackwardFilter(
        this->cuda_->cudnn(),
        &this->cuda_->one,
        this->input_desc_,
        this->input_->get_device_ptr(),
        this->output_desc_,
        this->grad_of_output_->get_device_ptr(),
        conv_desc_,
        conv_bwd_filter_algo_,
        device_workspace_,
        workspace_size_,
        &this->cuda_->zero,
        this->filter_desc_,
        this->grad_weights_->get_device_ptr()));

    // gradients of input data
    if (!this->gradient_stop_)
        checkCudnnErrors(cudnnConvolutionBackwardData(
            this->cuda_->cudnn(),
            &this->cuda_->one,
            this->filter_desc_,
            this->weights_->get_device_ptr(),
            this->output_desc_,
            grad_output->get_device_ptr(),
            conv_desc_,
            conv_bwd_data_algo_,
            device_workspace_,
            workspace_size_,
            &this->cuda_->zero,
            this->input_desc_,
            this->grad_of_input_->get_device_ptr()));

    if (DEBUG_CONV & 0x02) {
        std::cout << this->name_ << "[BACKWARD]" << std::endl;
        grad_output->print(this->name_ + "::gradients", true);
        this->grad_biases_->print(this->name_ + "gbias", true);
        this->grad_weights_->print(this->name_ + "gfilter", true);
        if (!this->gradient_stop_)
            this->grad_of_input_->print(this->name_ + "gdata", true);
    }

    if (DEBUG_CONV & 0x04) {
        grad_output->print(this->name_ + "::gradients", true);
        this->grad_biases_->print(this->name_ + "::gbias", true);
    }

    return this->grad_of_input_;
}

template <typename dtype> std::tuple<int, int> Conv2d<dtype>::calculate_fan_in_and_fan_out() {
    auto num_input_fmaps = this->in_channels_;
    auto num_output_fmaps = this->out_channels_;
    auto receptive_field_size = kernel_size_ * kernel_size_;
    return std::make_tuple(num_input_fmaps * receptive_field_size, num_output_fmaps * receptive_field_size);
}

template class Conv2d<float>;

template class Conv2d<double>;
