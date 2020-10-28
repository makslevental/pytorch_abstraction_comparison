#include <cudnn.h>
#include <iostream> // std::cout, std::ios
#include <sstream>  // std::ostringstream
#include <tuple>

template <typename T> using CudaPtr = T *;

typedef cudnnTensorDescriptor_t TensorDesc;
typedef cudnnFilterDescriptor_t FilterDesc;
typedef cudnnConvolutionDescriptor_t ConvDesc;

template <typename T, typename D> struct Tensor {
    // Handle to a previously initialized tensor descriptor
    D descriptor;
    // Data pointer to GPU memory associated with the tensor descriptor
    CudaPtr<T> ptr = nullptr;

    explicit Tensor(int size) {
        if constexpr (std::is_same<D, TensorDesc>{}) {
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&descriptor));
        } else if constexpr (std::is_same<D, FilterDesc>{}) {
            CUDNN_CHECK(cudnnCreateFilterDescriptor(&descriptor));
        } else if constexpr (std::is_same<D, ConvDesc>{}) {
            CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&descriptor));
        }
        CHECK_CUDA_ERRORS(cudaMalloc(&ptr, sizeof(T) * size));
    }
    ~Tensor() {
        if constexpr (std::is_same<D, TensorDesc>{}) {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(descriptor));
        } else if constexpr (std::is_same<D, FilterDesc>{}) {
            CUDNN_CHECK(cudnnDestroyFilterDescriptor(descriptor));
        } else if constexpr (std::is_same<D, ConvDesc>{}) {
            CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(descriptor));
        }
        CHECK_CUDA_ERRORS(cudaFree(ptr));
    }
};

typedef Tensor<float, TensorDesc> FloatTensor;
typedef Tensor<float, FilterDesc> FloatFilter;
typedef Tensor<float, ConvDesc> FloatConv;

struct ConvBiasLayer {
    int inChannels, outChannels, kernelSize;
    int inWidth, inHeight, outWidth, outHeight;
    int outputSize;
    int gpuid;
    int batchSize;
    bool backPropData;
    float alpha = 1.0f, beta = 0.0f;
    float *dBias;
    float *dConv;

    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
    cudnnTensorFormat_t tensorFormat;
    FloatTensor output, bias;
    FloatFilter filter;
    FloatConv conv;
    cudnnConvolutionFwdAlgo_t fwdAlgo;
    cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
    cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;

    ConvBiasLayer(
        int in_channels,
        int out_channels,
        int kernel_size,
        int in_width,
        int in_height,
        int gpuid,
        int batch_size,
        bool backprop_data = true,
        cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NCHW)
        : inChannels(in_channels), outChannels(out_channels), kernelSize(kernel_size),
          inWidth(in_width), inHeight(in_height), gpuid(gpuid), batchSize(batch_size),
          outputSize(in_channels * kernel_size * kernel_size * out_channels), output(outputSize),
          filter(outputSize), bias(out_channels), backPropData(backprop_data),
          outWidth(in_width - kernel_size + 1), outHeight(in_height - kernel_size + 1),
          conv(sizeof(float) * batchSize * outChannels * outHeight * outWidth),
          tensorFormat(tensorFormat) {

        CHECK_CUDA_ERRORS(cudaSetDevice(gpuid));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            /*tensorDesc*/ bias.descriptor,
            /*format*/ tensorFormat,
            /*dataType*/ CUDNN_DATA_FLOAT,
            /*n*/ 1,
            /*c*/ outChannels,
            /*h*/ 1,
            /*w*/ 1));
    }

    size_t SetFwdConvolutionTensors(const Tensor<float, TensorDesc> &input) {
        size_t sizeInBytes = 0;

        CHECK_CUDA_ERRORS(cudaSetDevice(gpuid));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            input.descriptor,
            tensorFormat,
            CUDNN_DATA_FLOAT,
            batchSize,
            inChannels,
            inHeight,
            inWidth));

        CUDNN_CHECK(cudnnSetFilter4dDescriptor(
            filter.descriptor,
            CUDNN_DATA_FLOAT,
            tensorFormat,
            outChannels,
            inChannels,
            kernelSize,
            kernelSize));

        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv.descriptor,
            /*pad_h*/ 0,
            /*pad_w*/ 0,
            /*u*/ 1, // uv is input (or output?) stride
            /*v*/ 1,
            /*dilation_h*/ 1,
            /*dilation_w*/ 1,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT));

        // Find dimension of convolution output
        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
            conv.descriptor,
            input.descriptor,
            filter.descriptor,
            &batchSize,
            &inChannels,
            &inHeight,
            &inWidth));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            output.descriptor,
            tensorFormat,
            CUDNN_DATA_FLOAT,
            batchSize,
            inChannels,
            inHeight,
            inWidth));
        int max_algos, returned_algo_count;
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &max_algos));
        cudnnConvolutionFwdAlgoPerf_t perfResults;
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            cudnnHandle,
            input.descriptor,
            filter.descriptor,
            conv.descriptor,
            output.descriptor,
            max_algos,
            &returned_algo_count,
            &perfResults));

        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cudnnHandle,
            input.descriptor,
            filter.descriptor,
            conv.descriptor,
            output.descriptor,
            fwdAlgo,
            &sizeInBytes));

        return sizeInBytes;
    }

    void Foward(const Tensor<float, TensorDesc> &input, void *workspace, size_t workspace_size) {
        CHECK_CUDA_ERRORS(cudaSetDevice(gpuid));

        CUDNN_CHECK(cudnnConvolutionForward(
            cudnnHandle,
            &alpha,
            input.descriptor,
            input.ptr,
            filter.descriptor,
            filter.ptr,
            conv.descriptor,
            fwdAlgo,
            workspace,
            workspace_size,
            &beta,
            output.descriptor,
            output.ptr));
        CUDNN_CHECK(cudnnAddTensor(
            cudnnHandle, &alpha, bias.descriptor, bias.ptr, &alpha, output.descriptor, output.ptr));
    }

    size_t SetBwdConvolutionTensors(const Tensor<float, TensorDesc> &input) {
        size_t sizeInBytes = 0, tmpsize = 0;

        // If backprop filter algorithm was requested
        int max_algos, returned_algo_count;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle, &max_algos));
        cudnnConvolutionBwdFilterAlgoPerf_t perfResults;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            cudnnHandle,
            input.descriptor,
            output.descriptor,
            conv.descriptor,
            filter.descriptor,
            max_algos,
            &returned_algo_count,
            &perfResults));

        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            cudnnHandle,
            input.descriptor,
            output.descriptor,
            conv.descriptor,
            filter.descriptor,
            bwdFilterAlgo,
            &tmpsize));

        sizeInBytes = std::max(sizeInBytes, tmpsize);

        // If backprop data algorithm was requested
        if (backPropData) {
            CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle, &max_algos));
            cudnnConvolutionBwdDataAlgoPerf_t perfResults;
            CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                cudnnHandle,
                filter.descriptor,
                output.descriptor,
                conv.descriptor,
                input.descriptor,
                max_algos,
                &returned_algo_count,
                &perfResults));

            CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnnHandle,
                filter.descriptor,
                output.descriptor,
                conv.descriptor,
                input.descriptor,
                bwdDataAlgo,
                &tmpsize));

            sizeInBytes = std::max(sizeInBytes, tmpsize);
        }

        return sizeInBytes;
    }

    void Backward(
        const Tensor<float, TensorDesc> &next,
        const Tensor<float, TensorDesc> &previous,
        const Tensor<float, TensorDesc> &dNext,
        const Tensor<float, TensorDesc> &dPrevious,
        void *workspace,
        size_t workspace_size) {
        CHECK_CUDA_ERRORS(cudaSetDevice(gpuid));

        CUDNN_CHECK(cudnnConvolutionBackwardBias(
            // this doesn't make sense (needs dNext output from outside
            cudnnHandle,
            &alpha,
            output.descriptor,
            dNext.ptr,
            &beta,
            bias.descriptor,
            dBias));

        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
            cudnnHandle,
            &alpha,
            previous.descriptor,
            previous.ptr,
            output.descriptor,
            dNext.ptr,
            conv.descriptor,
            bwdFilterAlgo,
            workspace,
            workspace_size,
            &beta,
            filter.descriptor,
            dConv));

        if (backPropData) {
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                cudnnHandle,
                &alpha,
                filter.descriptor,
                filter.ptr,
                output.descriptor,
                dNext.ptr,
                conv.descriptor,
                bwdDataAlgo,
                workspace,
                workspace_size,
                &beta,
                previous.descriptor,
                dConv));
        }
    }
};

// struct MaxPoolLayerParams {
//    int size, stride;
//    MaxPoolLayerParams(int size_, int stride_) : size(size_), stride(stride_) {}
//};
//
// struct FullyConnectedLayerParams {
//    int inputs, outputs;
//    std::vector<float> pneurons, pbias;
//
//    FullyConnectedLayerParams(int inputs_, int outputs_)
//        : outputs(outputs_), inputs(inputs_), pneurons(inputs_ * outputs_), pbias(outputs_) {}
//};
//
// struct BatchNormLayerParams {
//    cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
//    float alpha = 1.0f, beta = 0.0f;
//    double epsilon = 1e-5;
//};
