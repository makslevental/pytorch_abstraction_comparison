//
// Created by Maksim Levental on 10/29/20.
//

#include <layers/batch_norm_2d.cuh>

BatchNorm2d::BatchNorm2d(
    std::string name,
    double epsilon,
    double momentum,
    bool affine,
    bool track_running_stats,
    cudnnBatchNormMode_t mode)
    : epsilon_(epsilon), momentum_(momentum), affine_(affine),
      track_running_stats_(track_running_stats), mode_(mode) {

    name_ = std::move(name);
    checkCudnnErrors(cudnnCreateTensorDescriptor(&derived_bn_scale_bias_mean_var_desc_));
}
BatchNorm2d::~BatchNorm2d() = default;

Tensor<double> *BatchNorm2d::forward(Tensor<double> *input) {
    fwd_initialize(input);
    input_ = input;
    if (train_) {
        checkCudnnErrors(cudnnBatchNormalizationForwardTrainingEx(
            /*handle*/ cuda_->cudnn(),
            /*mode*/ mode_,
            /*bnOps*/ CUDNN_BATCHNORM_OPS_BN,
            /**alpha*/ &cuda_->one,
            /**beta*/ &cuda_->zero,
            /*xDesc*/ input_desc_,
            /**xData*/ input->get_device_ptr(),
            /*zDesc */ nullptr,  // z descriptor for BN-Add-Relu
            /**zData */ nullptr, // z for BN-Add-ReLU
            /*yDesc*/ output_desc_,
            /**yData*/ output_->get_device_ptr(),
            /*bnScaleBiasMeanVarDesc*/ derived_bn_scale_bias_mean_var_desc_,
            /**bnScaleData*/ weights_->get_device_ptr(),
            /**bnBiasData */ biases_->get_device_ptr(),
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
            /*handle*/ cuda_->cudnn(),
            /*mode*/ mode_,
            /**alpha*/ &cuda_->one,
            /**beta*/ &cuda_->zero,
            /*xDesc*/ input_desc_,
            /**x*/ input->get_device_ptr(),
            /*yDesc*/ output_desc_,
            /**y*/ output_->get_device_ptr(),
            /*bnScaleBiasMeanVarDesc*/ derived_bn_scale_bias_mean_var_desc_,
            /**bnScaleData*/ weights_->get_device_ptr(),
            /**bnBiasData */ biases_->get_device_ptr(),
            /**estimatedMean*/ running_mean_->get_device_ptr(),
            /**estimatedVariance*/ running_var_->get_device_ptr(),
            /*epsilon*/ epsilon_));
    }
    // will i need to clone this?
    return output_;
}
Tensor<double> *BatchNorm2d::backward(Tensor<double> *grad_output) {
    bwd_initialize(grad_output);
    checkCudnnErrors(cudnnBatchNormalizationBackwardEx(
        /*handle*/ cuda_->cudnn(),
        /*mode*/ mode_,
        /*bnOps*/ CUDNN_BATCHNORM_OPS_BN,
        /**alphaDataDiff*/ &cuda_->one,
        /**betaDataDiff*/ &cuda_->zero,
        /**alphaParamDiff*/ &cuda_->one,
        /**betaParamDiff*/ &cuda_->zero,
        /*xDesc*/ input_desc_,
        /**xData*/ input_->get_device_ptr(),
        /*yDesc*/ nullptr,
        /**yData*/ nullptr,
        /*dyDesc*/ grad_output->tensor_descriptor(),
        /**dyData*/ grad_output->get_device_ptr(),
        /*dzDesc*/ nullptr,
        /**dzData*/ nullptr,
        /*dxDesc*/ grad_input_->tensor_descriptor(),
        /**dxData*/ grad_input_->get_device_ptr(),
        /*dBnScaleBiasDesc*/ derived_bn_scale_bias_mean_var_desc_,
        /**bnScaleData*/ weights_->get_device_ptr(),
        /**bnBiasData*/ biases_->get_device_ptr(),
        /**dBnScaleData*/ grad_weights_->get_device_ptr(),
        /**dBnBiasData*/ grad_biases_->get_device_ptr(),
        /*epsilon*/ epsilon_,
        /**savedMean*/ save_mean_->get_device_ptr(),
        /**savedInvVariance*/ save_var_->get_device_ptr(),
        /*activationDesc*/ nullptr,
        /**workspace*/ device_workspace_,
        /*workSpaceSizeInBytes*/ workspace_size_,
        /**reserveSpace*/ device_reserve_space_,
        /*reserveSpaceSizeInBytes*/ reserve_size_));
    return grad_input_;
}

void BatchNorm2d::fwd_initialize(Tensor<double> *input) {
    // initialize weights and bias
    if (weights_ == nullptr) {
        if (mode_ == CUDNN_BATCHNORM_PER_ACTIVATION) {
            weights_ = new Tensor<double>(
                1, input->get_channels(), input->get_height(), input->get_width());
            biases_ = new Tensor<double>(
                1, input->get_channels(), input->get_height(), input->get_width());
        } else if (
            mode_ == CUDNN_BATCHNORM_SPATIAL || mode_ == CUDNN_BATCHNORM_SPATIAL_PERSISTENT) {
            weights_ = new Tensor<double>(1, input->get_channels());
            biases_ = new Tensor<double>(1, input->get_channels());
        } else {
            exit(EXIT_FAILURE);
        }
    }
    // initilaize input and output
    if (input_desc_ == nullptr || batch_size_ != input->get_batch_size()) {
        input_desc_ = input->tensor_descriptor();
        input_size_ = input->size();
        batch_size_ = input->get_batch_size();
        num_features_ = input->get_channels();
        if (track_running_stats_) {
            running_mean_ = new Tensor<double>(1, num_features_);
            running_var_ = new Tensor<double>(1, num_features_);
        }

        save_mean_ = new Tensor<double>(1, num_features_);
        save_var_ = new Tensor<double>(1, num_features_);

        if (output_ == nullptr) {
            output_ = new Tensor<double>(input->shape());
        } else {
            output_->reset(input->shape());
        }
        output_desc_ = output_->tensor_descriptor();

        checkCudnnErrors(cudnnDeriveBNTensorDescriptor(
            derived_bn_scale_bias_mean_var_desc_, input_desc_, mode_));

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
        }
    }
}

void BatchNorm2d::bwd_initialize(Tensor<double> *grad_output) {
    if (grad_weights_ == nullptr) {
        grad_weights_ = new Tensor<double>(weights_->shape());
        grad_biases_ = new Tensor<double>(biases_->shape());
    }
    Layer::bwd_initialize(grad_output);
}
void BatchNorm2d::set_workspace() {
    checkCudnnErrors(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        /*handle*/ cuda_->cudnn(),
        /*mode*/ mode_,
        /*bnOps*/ CUDNN_BATCHNORM_OPS_BN,
        /*xDesc*/ input_desc_,
        /*zDesc*/ input_desc_,
        /*yDesc*/ input_desc_,
        /*bnScaleBiasMeanVarDesc*/ derived_bn_scale_bias_mean_var_desc_,
        /*activationDesc*/ nullptr,
        /**sizeInBytes*/ &workspace_size_));
    size_t workspace_size;
    checkCudnnErrors(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        /*handle*/ cuda_->cudnn(),
        /*mode*/ mode_,
        /*bnOps*/ CUDNN_BATCHNORM_OPS_BN,
        /*xDesc*/ input_desc_,
        /*yDesc*/ input_desc_,
        /*dyDesc*/ input_desc_,
        /*dzDesc*/ nullptr,
        /*dxDesc*/ output_->tensor_descriptor(),
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
        cuda_->cudnn(), mode_, CUDNN_BATCHNORM_OPS_BN, nullptr, input_desc_, &reserve_size_));
    //    PRINT("reserve_size_ " << reserve_size_)
    if (reserve_size_ > 0) {
        if (device_reserve_space_ != nullptr)
            checkCudaErrors(cudaFree(device_reserve_space_));
        checkCudaErrors(cudaMalloc((void **)&device_reserve_space_, reserve_size_));
    }
}
