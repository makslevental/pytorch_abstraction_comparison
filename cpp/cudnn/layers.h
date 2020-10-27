struct ConvBiasLayer {
    int in_channels, out_channels, kernel_size;
    int in_width, in_height, out_width, out_height;

    std::vector<float> pconv, pbias;

    ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_, int in_w_, int in_h_)
        : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_), pbias(out_channels_) {
        in_channels = in_channels_;
        out_channels = out_channels_;
        kernel_size = kernel_size_;
        in_width = in_w_;
        in_height = in_h_;
        out_width = in_w_ - kernel_size_ + 1;
        out_height = in_h_ - kernel_size_ + 1;
    }
};

struct MaxPoolLayer {
    int size, stride;
    MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};

struct FullyConnectedLayer {
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_)
        : outputs(outputs_), inputs(inputs_), pneurons(inputs_ * outputs_), pbias(outputs_) {}
};

struct FullyConnectedLayer {
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_)
        : outputs(outputs_), inputs(inputs_), pneurons(inputs_ * outputs_), pbias(outputs_) {}
};

