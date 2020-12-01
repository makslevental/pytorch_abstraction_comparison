//
// Created by Maksim Levental on 10/29/20.
//

#include <layers/batch_norm_2d.cuh>

template <typename dtype>
BatchNorm2d<dtype>::BatchNorm2d(
    std::string name,
    double epsilon,
    double momentum,
    bool affine,
    bool track_running_stats,
    cudnnBatchNormMode_t mode)
    : epsilon_(epsilon), momentum_(momentum), affine_(affine),
      track_running_stats_(track_running_stats), mode_(mode) {

    this->name_ = std::move(name);
    checkCudnnErrors(cudnnCreateTensorDescriptor(&derived_bn_scale_bias_mean_var_desc_));
}

template <typename dtype> Tensor<dtype> *BatchNorm2d<dtype>::forward(Tensor<dtype> *input) {
    fwd_initialize(input);
    this->input_ = input;
    if (this->train_) {
        checkCudnnErrors(cudnnBatchNormalizationForwardTrainingEx(
            /*handle*/ this->cuda_->cudnn(),
            /*mode*/ mode_,
            /*bnOps*/ CUDNN_BATCHNORM_OPS_BN,
            /**alpha*/ &this->cuda_->one,
            /**beta*/ &this->cuda_->zero,
            /*xDesc*/ this->input_desc_,
            /**xData*/ input->get_device_ptr(),
            /*zDesc */ nullptr,  // z descriptor for BN-Add-Relu
            /**zData */ nullptr, // z for BN-Add-ReLU
            /*yDesc*/ this->output_desc_,
            /**yData*/ this->output_->get_device_ptr(),
            /*bnScaleBiasMeanVarDesc*/ derived_bn_scale_bias_mean_var_desc_,
            /**bnScaleData*/ this->weights_->get_device_ptr(),
            /**bnBiasData */ this->biases_->get_device_ptr(),
            /*exponentialAverageFactor*/ momentum_, // TODO: something about the first one having to
                                                    // be 1?
            /**resultRunningMeanData*/ running_mean_->get_device_ptr(),
            /**resultRunningVarianceData*/ running_var_->get_device_ptr(),
            /*epsilon*/ epsilon_,
            /**saveMean*/ save_mean_->get_device_ptr(),
            /**saveInvVariance*/ save_var_->get_device_ptr(),
            /*activationDesc */ nullptr,
            /**workspace*/ device_workspace_,
            /*workSpaceSizeInBytes*/ workspace_size_,
            /**reserveSpace*/ device_reserve_space_,
            /*reserveSpaceSizeInBytes*/ reserve_size_));
    } else {
        checkCudnnErrors(cudnnBatchNormalizationForwardInference(
            /*handle*/ this->cuda_->cudnn(),
            /*mode*/ mode_,
            /**alpha*/ &this->cuda_->one,
            /**beta*/ &this->cuda_->zero,
            /*xDesc*/ this->input_desc_,
            /**x*/ input->get_device_ptr(),
            /*yDesc*/ this->output_desc_,
            /**y*/ this->output_->get_device_ptr(),
            /*bnScaleBiasMeanVarDesc*/ derived_bn_scale_bias_mean_var_desc_,
            /**bnScaleData*/ this->weights_->get_device_ptr(),
            /**bnBiasData */ this->biases_->get_device_ptr(),
            /**estimatedMean*/ running_mean_->get_device_ptr(),
            /**estimatedVariance*/ running_var_->get_device_ptr(),
            /*epsilon*/ epsilon_));
    }
    // will i need to clone this?
    return this->output_;
}
template <typename dtype> Tensor<dtype> *BatchNorm2d<dtype>::backward(Tensor<dtype> *grad_output) {
    bwd_initialize(grad_output);
    checkCudnnErrors(cudnnBatchNormalizationBackwardEx(
        /*handle*/ this->cuda_->cudnn(),
        /*mode*/ mode_,
        /*bnOps*/ CUDNN_BATCHNORM_OPS_BN,
        /**alphaDataDiff*/ &this->cuda_->one,
        /**betaDataDiff*/ &this->cuda_->zero,
        /**alphaParamDiff*/ &this->cuda_->one,
        /**betaParamDiff*/ &this->cuda_->zero,
        /*xDesc*/ this->input_desc_,
        /**xData*/ this->input_->get_device_ptr(),
        /*yDesc*/ nullptr,
        /**yData*/ nullptr,
        /*dyDesc*/ grad_output->tensor_descriptor(),
        /**dyData*/ grad_output->get_device_ptr(),
        /*dzDesc*/ nullptr,
        /**dzData*/ nullptr,
        /*dxDesc*/ this->grad_of_input_->tensor_descriptor(),
        /**dxData*/ this->grad_of_input_->get_device_ptr(),
        /*dBnScaleBiasDesc*/ derived_bn_scale_bias_mean_var_desc_,
        /**bnScaleData*/ this->weights_->get_device_ptr(),
        /**bnBiasData*/ this->biases_->get_device_ptr(),
        /**dBnScaleData*/ this->grad_weights_->get_device_ptr(),
        /**dBnBiasData*/ this->grad_biases_->get_device_ptr(),
        /*epsilon*/ epsilon_,
        /**savedMean*/ save_mean_->get_device_ptr(),
        /**savedInvVariance*/ save_var_->get_device_ptr(),
        /*activationDesc*/ nullptr,
        /**workspace*/ device_workspace_,
        /*workSpaceSizeInBytes*/ workspace_size_,
        /**reserveSpace*/ device_reserve_space_,
        /*reserveSpaceSizeInBytes*/ reserve_size_));

    return this->grad_of_input_;
}

template <typename dtype> void BatchNorm2d<dtype>::fwd_initialize(Tensor<dtype> *input) {
    // initialize weights and bias
    if (this->weights_ == nullptr) {
        if (mode_ == CUDNN_BATCHNORM_PER_ACTIVATION) {
            this->weights_ = new Tensor<dtype>(
                1, input->get_channels(), input->get_height(), input->get_width());
            this->biases_ = new Tensor<dtype>(
                1, input->get_channels(), input->get_height(), input->get_width());
        } else if (
            mode_ == CUDNN_BATCHNORM_SPATIAL || mode_ == CUDNN_BATCHNORM_SPATIAL_PERSISTENT) {
            this->weights_ = new Tensor<dtype>(1, input->get_channels());
            this->biases_ = new Tensor<dtype>(1, input->get_channels());
        } else {
            exit(EXIT_FAILURE);
        }
    }
    // initilaize input and output
    if (this->input_desc_ == nullptr || this->batch_size_ != input->get_batch_size()) {
        this->input_desc_ = input->tensor_descriptor();
        this->input_size_ = input->size();
        this->batch_size_ = input->get_batch_size();
        num_features_ = input->get_channels();
        if (track_running_stats_) {
            running_mean_ = new Tensor<dtype>(1, num_features_);
            running_var_ = new Tensor<dtype>(1, num_features_);
        }

        save_mean_ = new Tensor<dtype>(1, num_features_);
        save_var_ = new Tensor<dtype>(1, num_features_);

        if (this->output_ == nullptr) {
            this->output_ = new Tensor<dtype>(input->shape());
        } else {
            this->output_->reset(input->shape());
        }
        this->output_desc_ = this->output_->tensor_descriptor();

        checkCudnnErrors(cudnnDeriveBNTensorDescriptor(
            derived_bn_scale_bias_mean_var_desc_, this->input_desc_, mode_));

        // initialize workspace for cudnn
        set_workspace();

        // initialize weights
        if (this->load_pretrain_ && !this->freeze_) {
            if (this->load_parameter()) {
                std::cout << "error occurred loading params batch norm" << std::endl;
                exit(EXIT_FAILURE);
            }
        } else if (!this->freeze_) {
            this->weights_->one_out();
            this->biases_->zero_out();
//            this->weights_->print("weights", true);
//            this->biases_->print("biases", true);
        }
    }
}

template <typename dtype> void BatchNorm2d<dtype>::bwd_initialize(Tensor<dtype> *grad_output) {
    if (this->grad_weights_ == nullptr) {
        this->grad_weights_ = new Tensor<dtype>(this->weights_->shape());
        this->grad_biases_ = new Tensor<dtype>(this->biases_->shape());
    }
    Layer<dtype>::bwd_initialize(grad_output);
}
template <typename dtype> void BatchNorm2d<dtype>::set_workspace() {
    checkCudnnErrors(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        /*handle*/ this->cuda_->cudnn(),
        /*mode*/ mode_,
        /*bnOps*/ CUDNN_BATCHNORM_OPS_BN,
        /*xDesc*/ this->input_desc_,
        /*zDesc*/ this->input_desc_,
        /*yDesc*/ this->input_desc_,
        /*bnScaleBiasMeanVarDesc*/ derived_bn_scale_bias_mean_var_desc_,
        /*activationDesc*/ nullptr,
        /**sizeInBytes*/ &workspace_size_));
    size_t workspace_size;
    checkCudnnErrors(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        /*handle*/ this->cuda_->cudnn(),
        /*mode*/ mode_,
        /*bnOps*/ CUDNN_BATCHNORM_OPS_BN,
        /*xDesc*/ this->input_desc_,
        /*yDesc*/ this->input_desc_,
        /*dyDesc*/ this->input_desc_,
        /*dzDesc*/ nullptr,
        /*dxDesc*/ this->output_->tensor_descriptor(),
        /*dBnScaleBiasDesc*/ derived_bn_scale_bias_mean_var_desc_,
        /*activationDesc*/ nullptr,
        &workspace_size));

    // TODO: why are these still zero???
    workspace_size_ = std::max(workspace_size_, workspace_size);
    //    PRINT("workspace_size_ " << workspace_size_)
    if (workspace_size_ > 0) {
        if (device_workspace_ != nullptr)
            checkCudaErrors(cudaFree(device_workspace_));
        checkCudaErrors(cudaMalloc((void **)&device_workspace_, workspace_size_));
    }

    checkCudnnErrors(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        this->cuda_->cudnn(),
        mode_,
        CUDNN_BATCHNORM_OPS_BN,
        nullptr,
        this->input_desc_,
        &reserve_size_));
    //    PRINT("reserve_size_ " << reserve_size_)
    if (reserve_size_ > 0) {
        if (device_reserve_space_ != nullptr)
            checkCudaErrors(cudaFree(device_reserve_space_));
        checkCudaErrors(cudaMalloc((void **)&device_reserve_space_, reserve_size_));
    }
}

template class BatchNorm2d<float>;
template class BatchNorm2d<double>;
