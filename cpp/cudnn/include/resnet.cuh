//
// Created by Maksim Levental on 10/29/20.
//

// TODO: the downsampling steps are wrong
#include "layers.cuh"
#include "network.h"

class ResNet50 : Network {
public:
    ResNet50() {
        this->conv1 =
            new Conv2d("conv1", /*out_channels*/ 64, /*kernel*/ 7, /*stride*/ 2, /*padding*/ 3);
        this->conv1->set_gradient_stop();
        this->bn1 = new BatchNorm2d("bn1");
        this->relu1 = new Activation("relu1", CUDNN_ACTIVATION_RELU);
        this->pool1 =
            new Pooling("pool1", /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1, CUDNN_POOLING_MAX);
        // layer 1
        // bottleneck 1

        this->conv2 =
            new Conv2d("conv2", /*out_channels*/ 64, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn2 = new BatchNorm2d("bn2");
        this->conv3 =
            new Conv2d("conv3", /*out_channels*/ 64, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn3 = new BatchNorm2d("bn3");
        this->conv4 =
            new Conv2d("conv4", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn4 = new BatchNorm2d("bn4");
        this->relu2 = new Activation("relu2", CUDNN_ACTIVATION_RELU);
        // downsample

        this->conv5 =
            new Conv2d("conv5", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn5 = new BatchNorm2d("bn5");
        // bottleneck 2

        this->conv6 =
            new Conv2d("conv6", /*out_channels*/ 64, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn6 = new BatchNorm2d("bn6");
        this->conv7 =
            new Conv2d("conv7", /*out_channels*/ 64, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn7 = new BatchNorm2d("bn7");
        this->conv8 =
            new Conv2d("conv8", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn8 = new BatchNorm2d("bn8");
        this->relu3 = new Activation("relu3", CUDNN_ACTIVATION_RELU);
        // bottleneck 3

        this->conv9 =
            new Conv2d("conv9", /*out_channels*/ 64, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn9 = new BatchNorm2d("bn9");
        this->conv10 =
            new Conv2d("conv10", /*out_channels*/ 64, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn10 = new BatchNorm2d("bn10");
        this->conv11 =
            new Conv2d("conv11", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn11 = new BatchNorm2d("bn11");
        this->relu4 = new Activation("relu4", CUDNN_ACTIVATION_RELU);
        // layer 2
        // bottleneck 1

        this->conv12 =
            new Conv2d("conv12", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn12 = new BatchNorm2d("bn12");
        this->conv13 =
            new Conv2d("conv13", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1);
        this->bn13 = new BatchNorm2d("bn13");
        this->conv14 =
            new Conv2d("conv14", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn14 = new BatchNorm2d("bn14");
        this->relu5 = new Activation("relu5", CUDNN_ACTIVATION_RELU);
        // downsample

        this->conv15 =
            new Conv2d("conv15", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 2, /*padding*/ 0);
        this->bn15 = new BatchNorm2d("bn15");
        // bottleneck 2

        this->conv16 =
            new Conv2d("conv16", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn16 = new BatchNorm2d("bn16");
        this->conv17 =
            new Conv2d("conv17", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn17 = new BatchNorm2d("bn17");
        this->conv18 =
            new Conv2d("conv18", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn18 = new BatchNorm2d("bn18");
        this->relu6 = new Activation("relu6", CUDNN_ACTIVATION_RELU);
        // bottleneck 3

        this->conv19 =
            new Conv2d("conv19", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn19 = new BatchNorm2d("bn19");
        this->conv20 =
            new Conv2d("conv20", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn20 = new BatchNorm2d("bn20");
        this->conv21 =
            new Conv2d("conv21", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn21 = new BatchNorm2d("bn21");
        this->relu7 = new Activation("relu7", CUDNN_ACTIVATION_RELU);
        // bottleneck 4

        this->conv22 =
            new Conv2d("conv22", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn22 = new BatchNorm2d("bn22");
        this->conv23 =
            new Conv2d("conv23", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn23 = new BatchNorm2d("bn23");
        this->conv24 =
            new Conv2d("conv24", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn24 = new BatchNorm2d("bn24");
        this->relu8 = new Activation("relu8", CUDNN_ACTIVATION_RELU);
        // layer 3
        // bottleneck 1

        this->conv25 =
            new Conv2d("conv25", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn25 = new BatchNorm2d("bn25");
        this->conv26 =
            new Conv2d("conv26", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1);
        this->bn26 = new BatchNorm2d("bn26");
        this->conv27 =
            new Conv2d("conv27", /*out_channels*/ 1024, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn27 = new BatchNorm2d("bn27");
        this->relu9 = new Activation("relu9", CUDNN_ACTIVATION_RELU);
        // downsample

        this->conv28 =
            new Conv2d("conv28", /*out_channels*/ 1024, /*kernel*/ 1, /*stride*/ 2, /*padding*/ 0);
        this->bn28 = new BatchNorm2d("bn28");
        // bottleneck 2

        this->conv29 =
            new Conv2d("conv29", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn29 = new BatchNorm2d("bn29");
        this->conv30 =
            new Conv2d("conv30", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn30 = new BatchNorm2d("bn30");
        this->conv31 =
            new Conv2d("conv31", /*out_channels*/ 1024, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn31 = new BatchNorm2d("bn31");
        this->relu10 = new Activation("relu10", CUDNN_ACTIVATION_RELU);
        // bottleneck 3

        this->conv32 =
            new Conv2d("conv32", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn32 = new BatchNorm2d("bn32");
        this->conv33 =
            new Conv2d("conv33", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn33 = new BatchNorm2d("bn33");
        this->conv34 =
            new Conv2d("conv34", /*out_channels*/ 1024, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn34 = new BatchNorm2d("bn34");
        this->relu11 = new Activation("relu11", CUDNN_ACTIVATION_RELU);
        // bottleneck 4

        this->conv35 =
            new Conv2d("conv35", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn35 = new BatchNorm2d("bn35");
        this->conv36 =
            new Conv2d("conv36", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn36 = new BatchNorm2d("bn36");
        this->conv37 =
            new Conv2d("conv37", /*out_channels*/ 1024, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn37 = new BatchNorm2d("bn37");
        this->relu12 = new Activation("relu12", CUDNN_ACTIVATION_RELU);
        // bottleneck 5

        this->conv38 =
            new Conv2d("conv38", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn38 = new BatchNorm2d("bn38");
        this->conv39 =
            new Conv2d("conv39", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn39 = new BatchNorm2d("bn39");
        this->conv40 =
            new Conv2d("conv40", /*out_channels*/ 1024, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn40 = new BatchNorm2d("bn40");
        this->relu13 = new Activation("relu13", CUDNN_ACTIVATION_RELU);
        // layer 4
        // bottleneck 1

        this->conv41 =
            new Conv2d("conv41", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn41 = new BatchNorm2d("bn41");
        this->conv42 =
            new Conv2d("conv42", /*out_channels*/ 512, /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1);
        this->bn42 = new BatchNorm2d("bn42");
        this->conv43 =
            new Conv2d("conv43", /*out_channels*/ 2048, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn43 = new BatchNorm2d("bn43");
        this->relu14 = new Activation("relu14", CUDNN_ACTIVATION_RELU);
        // downsample

        this->conv44 =
            new Conv2d("conv44", /*out_channels*/ 2048, /*kernel*/ 1, /*stride*/ 2, /*padding*/ 0);
        this->bn44 = new BatchNorm2d("bn44");
        // bottleneck 2

        this->conv45 =
            new Conv2d("conv45", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn45 = new BatchNorm2d("bn45");
        this->conv46 =
            new Conv2d("conv46", /*out_channels*/ 512, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn46 = new BatchNorm2d("bn46");
        this->conv47 =
            new Conv2d("conv47", /*out_channels*/ 2048, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn47 = new BatchNorm2d("bn47");
        this->relu15 = new Activation("relu15", CUDNN_ACTIVATION_RELU);
        // bottleneck 3

        this->conv48 =
            new Conv2d("conv48", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn48 = new BatchNorm2d("bn48");
        this->conv49 =
            new Conv2d("conv49", /*out_channels*/ 512, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1);
        this->bn49 = new BatchNorm2d("bn49");
        this->conv50 =
            new Conv2d("conv50", /*out_channels*/ 2048, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0);
        this->bn50 = new BatchNorm2d("bn50");
        this->relu16 = new Activation("relu16", CUDNN_ACTIVATION_RELU);
        this->pool2 =
            new Pooling("pool2", /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0, CUDNN_POOLING_MAX);
        this->dense1 = new Dense("dense1", 10);
        this->softmax1 = new Softmax("softmax1");
    }
    Tensor<float> *forward(Tensor<float> *input) override {
        auto output = this->conv1->forward(input);
        output = this->bn1->forward(output);
        output = this->relu1->forward(output);

        output = this->pool1->forward(output);

        // layer 1
        // bottleneck 1

        output = this->conv2->forward(output);
        output = this->bn2->forward(output);

        output = this->conv3->forward(output);
        output = this->bn3->forward(output);

        output = this->conv4->forward(output);
        output = this->bn4->forward(output);
        output = this->relu2->forward(output);

        // downsample

        output = this->conv5->forward(output);
        output = this->bn5->forward(output);

        // bottleneck 2

        output = this->conv6->forward(output);
        output = this->bn6->forward(output);

        output = this->conv7->forward(output);
        output = this->bn7->forward(output);

        output = this->conv8->forward(output);
        output = this->bn8->forward(output);
        output = this->relu3->forward(output);

        // bottleneck 3

        output = this->conv9->forward(output);
        output = this->bn9->forward(output);

        output = this->conv10->forward(output);
        output = this->bn10->forward(output);

        output = this->conv11->forward(output);
        output = this->bn11->forward(output);
        output = this->relu4->forward(output);

        // layer 2
        // bottleneck 1

        output = this->conv12->forward(output);
        output = this->bn12->forward(output);

        output = this->conv13->forward(output);
        output = this->bn13->forward(output);

        output = this->conv14->forward(output);
        output = this->bn14->forward(output);
        output = this->relu5->forward(output);

        // downsample

        output = this->conv15->forward(output);
        output = this->bn15->forward(output);

        // bottleneck 2

        output = this->conv16->forward(output);
        output = this->bn16->forward(output);

        output = this->conv17->forward(output);
        output = this->bn17->forward(output);

        output = this->conv18->forward(output);
        output = this->bn18->forward(output);
        output = this->relu6->forward(output);

        // bottleneck 3

        output = this->conv19->forward(output);
        output = this->bn19->forward(output);

        output = this->conv20->forward(output);
        output = this->bn20->forward(output);

        output = this->conv21->forward(output);
        output = this->bn21->forward(output);
        output = this->relu7->forward(output);

        // bottleneck 4

        output = this->conv22->forward(output);
        output = this->bn22->forward(output);

        output = this->conv23->forward(output);
        output = this->bn23->forward(output);

        output = this->conv24->forward(output);
        output = this->bn24->forward(output);
        output = this->relu8->forward(output);

        // layer 3
        // bottleneck 1

        output = this->conv25->forward(output);
        output = this->bn25->forward(output);

        output = this->conv26->forward(output);
        output = this->bn26->forward(output);
        output = this->conv27->forward(output);
        output = this->bn27->forward(output);
        output = this->relu9->forward(output);

        // downsample

        output = this->conv28->forward(output);
        output = this->bn28->forward(output);

        // bottleneck 2

        output = this->conv29->forward(output);
        output = this->bn29->forward(output);

        output = this->conv30->forward(output);
        output = this->bn30->forward(output);
        output = this->conv31->forward(output);
        output = this->bn31->forward(output);
        output = this->relu10->forward(output);

        // bottleneck 3

        output = this->conv32->forward(output);
        output = this->bn32->forward(output);

        output = this->conv33->forward(output);
        output = this->bn33->forward(output);
        output = this->conv34->forward(output);
        output = this->bn34->forward(output);
        output = this->relu11->forward(output);

        // bottleneck 4

        output = this->conv35->forward(output);
        output = this->bn35->forward(output);

        output = this->conv36->forward(output);
        output = this->bn36->forward(output);
        output = this->conv37->forward(output);
        output = this->bn37->forward(output);
        output = this->relu12->forward(output);

        // bottleneck 5

        output = this->conv38->forward(output);
        output = this->bn38->forward(output);

        output = this->conv39->forward(output);
        output = this->bn39->forward(output);
        output = this->conv40->forward(output);
        output = this->bn40->forward(output);
        output = this->relu13->forward(output);

        // layer 4
        // bottleneck 1

        output = this->conv41->forward(output);
        output = this->bn41->forward(output);

        output = this->conv42->forward(output);
        output = this->bn42->forward(output);
        output = this->conv43->forward(output);
        output = this->bn43->forward(output);
        output = this->relu14->forward(output);

        // downsample

        output = this->conv44->forward(output);
        output = this->bn44->forward(output);

        // bottleneck 2

        output = this->conv45->forward(output);
        output = this->bn45->forward(output);

        output = this->conv46->forward(output);
        output = this->bn46->forward(output);
        output = this->conv47->forward(output);
        output = this->bn47->forward(output);
        output = this->relu15->forward(output);

        // bottleneck 3

        output = this->conv48->forward(output);
        output = this->bn48->forward(output);

        output = this->conv49->forward(output);
        output = this->bn49->forward(output);
        output = this->conv50->forward(output);
        output = this->bn50->forward(output);
        output = this->relu16->forward(output);

        output = this->pool2->forward(output);
        output = this->dense1->forward(output);
        output = this->softmax1->forward(output);

        return output;
    }
    void backward(Tensor<float> *target) override {
        auto gradient = this->softmax1->backward(target);
        gradient = this->dense1->backward(gradient);
        gradient = this->pool2->backward(gradient);

        gradient = this->relu16->backward(gradient);
        gradient = this->bn50->backward(gradient);
        gradient = this->conv50->backward(gradient);
        gradient = this->bn49->backward(gradient);
        gradient = this->conv49->backward(gradient);

        gradient = this->bn48->backward(gradient);
        gradient = this->conv48->backward(gradient);

        // bottleneck 3

        gradient = this->relu15->backward(gradient);
        gradient = this->bn47->backward(gradient);
        gradient = this->conv47->backward(gradient);
        gradient = this->bn46->backward(gradient);
        gradient = this->conv46->backward(gradient);

        gradient = this->bn45->backward(gradient);
        gradient = this->conv45->backward(gradient);

        // bottleneck 2

        gradient = this->bn44->backward(gradient);
        gradient = this->conv44->backward(gradient);

        // downsample

        gradient = this->relu14->backward(gradient);
        gradient = this->bn43->backward(gradient);
        gradient = this->conv43->backward(gradient);
        gradient = this->bn42->backward(gradient);
        gradient = this->conv42->backward(gradient);

        gradient = this->bn41->backward(gradient);
        gradient = this->conv41->backward(gradient);

        // bottleneck 1
        // layer 4

        gradient = this->relu13->backward(gradient);
        gradient = this->bn40->backward(gradient);
        gradient = this->conv40->backward(gradient);
        gradient = this->bn39->backward(gradient);
        gradient = this->conv39->backward(gradient);

        gradient = this->bn38->backward(gradient);
        gradient = this->conv38->backward(gradient);

        // bottleneck 5

        gradient = this->relu12->backward(gradient);
        gradient = this->bn37->backward(gradient);
        gradient = this->conv37->backward(gradient);
        gradient = this->bn36->backward(gradient);
        gradient = this->conv36->backward(gradient);

        gradient = this->bn35->backward(gradient);
        gradient = this->conv35->backward(gradient);

        // bottleneck 4

        gradient = this->relu11->backward(gradient);
        gradient = this->bn34->backward(gradient);
        gradient = this->conv34->backward(gradient);
        gradient = this->bn33->backward(gradient);
        gradient = this->conv33->backward(gradient);

        gradient = this->bn32->backward(gradient);
        gradient = this->conv32->backward(gradient);

        // bottleneck 3

        gradient = this->relu10->backward(gradient);
        gradient = this->bn31->backward(gradient);
        gradient = this->conv31->backward(gradient);
        gradient = this->bn30->backward(gradient);
        gradient = this->conv30->backward(gradient);

        gradient = this->bn29->backward(gradient);
        gradient = this->conv29->backward(gradient);

        // bottleneck 2

        gradient = this->bn28->backward(gradient);
        gradient = this->conv28->backward(gradient);

        // downsample

        gradient = this->relu9->backward(gradient);
        gradient = this->bn27->backward(gradient);
        gradient = this->conv27->backward(gradient);
        gradient = this->bn26->backward(gradient);
        gradient = this->conv26->backward(gradient);

        gradient = this->bn25->backward(gradient);
        gradient = this->conv25->backward(gradient);

        // bottleneck 1
        // layer 3

        gradient = this->relu8->backward(gradient);
        gradient = this->bn24->backward(gradient);
        gradient = this->conv24->backward(gradient);

        gradient = this->bn23->backward(gradient);
        gradient = this->conv23->backward(gradient);

        gradient = this->bn22->backward(gradient);
        gradient = this->conv22->backward(gradient);

        // bottleneck 4

        gradient = this->relu7->backward(gradient);
        gradient = this->bn21->backward(gradient);
        gradient = this->conv21->backward(gradient);

        gradient = this->bn20->backward(gradient);
        gradient = this->conv20->backward(gradient);

        gradient = this->bn19->backward(gradient);
        gradient = this->conv19->backward(gradient);

        // bottleneck 3

        gradient = this->relu6->backward(gradient);
        gradient = this->bn18->backward(gradient);
        gradient = this->conv18->backward(gradient);

        gradient = this->bn17->backward(gradient);
        gradient = this->conv17->backward(gradient);

        gradient = this->bn16->backward(gradient);
        gradient = this->conv16->backward(gradient);

        // bottleneck 2

        gradient = this->bn15->backward(gradient);
        gradient = this->conv15->backward(gradient);

        // downsample

        gradient = this->relu5->backward(gradient);
        gradient = this->bn14->backward(gradient);
        gradient = this->conv14->backward(gradient);

        gradient = this->bn13->backward(gradient);
        gradient = this->conv13->backward(gradient);

        gradient = this->bn12->backward(gradient);
        gradient = this->conv12->backward(gradient);

        // bottleneck 1
        // layer 2

        gradient = this->relu4->backward(gradient);
        gradient = this->bn11->backward(gradient);
        gradient = this->conv11->backward(gradient);

        gradient = this->bn10->backward(gradient);
        gradient = this->conv10->backward(gradient);

        gradient = this->bn9->backward(gradient);
        gradient = this->conv9->backward(gradient);

        // bottleneck 3

        gradient = this->relu3->backward(gradient);
        gradient = this->bn8->backward(gradient);
        gradient = this->conv8->backward(gradient);

        gradient = this->bn7->backward(gradient);
        gradient = this->conv7->backward(gradient);

        gradient = this->bn6->backward(gradient);
        gradient = this->conv6->backward(gradient);

        // bottleneck 2

        gradient = this->bn5->backward(gradient);
        gradient = this->conv5->backward(gradient);

        // downsample

        gradient = this->relu2->backward(gradient);
        gradient = this->bn4->backward(gradient);
        gradient = this->conv4->backward(gradient);

        gradient = this->bn3->backward(gradient);
        gradient = this->conv3->backward(gradient);

        gradient = this->bn2->backward(gradient);
        gradient = this->conv2->backward(gradient);

        // bottleneck 1
        // layer 1

        gradient = this->pool1->backward(gradient);

        gradient = this->relu1->backward(gradient);
        gradient = this->bn1->backward(gradient);
        gradient = this->conv1->backward(gradient);
    }

private:
    Conv2d *conv1;
    Conv2d *conv2;
    Conv2d *conv3;
    Conv2d *conv4;
    Conv2d *conv5;
    Conv2d *conv6;
    Conv2d *conv7;
    Conv2d *conv8;
    Conv2d *conv9;
    Conv2d *conv10;
    Conv2d *conv11;
    Conv2d *conv12;
    Conv2d *conv13;
    Conv2d *conv14;
    Conv2d *conv15;
    Conv2d *conv16;
    Conv2d *conv17;
    Conv2d *conv18;
    Conv2d *conv19;
    Conv2d *conv20;
    Conv2d *conv21;
    Conv2d *conv22;
    Conv2d *conv23;
    Conv2d *conv24;
    Conv2d *conv25;
    Conv2d *conv26;
    Conv2d *conv27;
    Conv2d *conv28;
    Conv2d *conv29;
    Conv2d *conv30;
    Conv2d *conv31;
    Conv2d *conv32;
    Conv2d *conv33;
    Conv2d *conv34;
    Conv2d *conv35;
    Conv2d *conv36;
    Conv2d *conv37;
    Conv2d *conv38;
    Conv2d *conv39;
    Conv2d *conv40;
    Conv2d *conv41;
    Conv2d *conv42;
    Conv2d *conv43;
    Conv2d *conv44;
    Conv2d *conv45;
    Conv2d *conv46;
    Conv2d *conv47;
    Conv2d *conv48;
    Conv2d *conv49;
    Conv2d *conv50;
    BatchNorm2d *bn1;
    BatchNorm2d *bn2;
    BatchNorm2d *bn3;
    BatchNorm2d *bn4;
    BatchNorm2d *bn5;
    BatchNorm2d *bn6;
    BatchNorm2d *bn7;
    BatchNorm2d *bn8;
    BatchNorm2d *bn9;
    BatchNorm2d *bn10;
    BatchNorm2d *bn11;
    BatchNorm2d *bn12;
    BatchNorm2d *bn13;
    BatchNorm2d *bn14;
    BatchNorm2d *bn15;
    BatchNorm2d *bn16;
    BatchNorm2d *bn17;
    BatchNorm2d *bn18;
    BatchNorm2d *bn19;
    BatchNorm2d *bn20;
    BatchNorm2d *bn21;
    BatchNorm2d *bn22;
    BatchNorm2d *bn23;
    BatchNorm2d *bn24;
    BatchNorm2d *bn25;
    BatchNorm2d *bn26;
    BatchNorm2d *bn27;
    BatchNorm2d *bn28;
    BatchNorm2d *bn29;
    BatchNorm2d *bn30;
    BatchNorm2d *bn31;
    BatchNorm2d *bn32;
    BatchNorm2d *bn33;
    BatchNorm2d *bn34;
    BatchNorm2d *bn35;
    BatchNorm2d *bn36;
    BatchNorm2d *bn37;
    BatchNorm2d *bn38;
    BatchNorm2d *bn39;
    BatchNorm2d *bn40;
    BatchNorm2d *bn41;
    BatchNorm2d *bn42;
    BatchNorm2d *bn43;
    BatchNorm2d *bn44;
    BatchNorm2d *bn45;
    BatchNorm2d *bn46;
    BatchNorm2d *bn47;
    BatchNorm2d *bn48;
    BatchNorm2d *bn49;
    BatchNorm2d *bn50;
    Activation *relu1;
    Activation *relu2;
    Activation *relu3;
    Activation *relu4;
    Activation *relu5;
    Activation *relu6;
    Activation *relu7;
    Activation *relu8;
    Activation *relu9;
    Activation *relu10;
    Activation *relu11;
    Activation *relu12;
    Activation *relu13;
    Activation *relu14;
    Activation *relu15;
    Activation *relu16;
    Pooling *pool1;
    Pooling *pool2;
    Dense *dense1;
    Softmax *softmax1;
};

Network make_resnet50() {
    Network model;

    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 64, /*kernel*/ 7, /*stride*/ 2, /*padding*/ 3));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(new Activation("relu1", CUDNN_ACTIVATION_RELU));
    model.add_layer(
        new Pooling("pool1", /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1, CUDNN_POOLING_MAX));

    // layer 1
    // bottleneck 1
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 64, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 64, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2d("conv3", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // downsample
    model.add_layer(
        new Conv2d("0", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("1"));

    // bottleneck 2
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 64, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 64, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2d("conv3", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 3
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 64, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 64, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2d("conv3", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // layer 2
    // bottleneck 1
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2d("conv3", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // downsample
    model.add_layer(
        new Conv2d("0", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 2, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("1"));

    // bottleneck 2
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2d("conv3", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 3
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2d("conv3", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 4
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 128, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 128, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(
        new Conv2d("conv3", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // layer 3
    // bottleneck 1
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2d(
        "conv3",
        /*out_channels*/ 1024,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // downsample
    model.add_layer(
        new Conv2d("0", /*out_channels*/ 1024, /*kernel*/ 1, /*stride*/ 2, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("1"));

    // bottleneck 2
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2d(
        "conv3",
        /*out_channels*/ 1024,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 3
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2d(
        "conv3",
        /*out_channels*/ 1024,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 4
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2d(
        "conv3",
        /*out_channels*/ 1024,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 5
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 256, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 256, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2d(
        "conv3",
        /*out_channels*/ 1024,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // layer 4
    // bottleneck 1
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 512, /*kernel*/ 3, /*stride*/ 2, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2d(
        "conv3",
        /*out_channels*/ 2048,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // downsample
    model.add_layer(
        new Conv2d("0", /*out_channels*/ 2048, /*kernel*/ 1, /*stride*/ 2, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("1"));

    // bottleneck 2
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 512, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2d(
        "conv3",
        /*out_channels*/ 2048,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    // bottleneck 3
    model.add_layer(
        new Conv2d("conv1", /*out_channels*/ 512, /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0));
    model.add_layer(new BatchNorm2d("bn1"));
    model.add_layer(
        new Conv2d("conv2", /*out_channels*/ 512, /*kernel*/ 3, /*stride*/ 1, /*padding*/ 1));
    model.add_layer(new BatchNorm2d("bn2"));
    model.add_layer(new Conv2d(
        "conv3",
        /*out_channels*/ 2048,
        /*kernel*/ 1,
        /*stride*/ 1, /*padding*/
        0));
    model.add_layer(new BatchNorm2d("bn3"));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));

    model.add_layer(
        new Pooling("pool1", /*kernel*/ 1, /*stride*/ 1, /*padding*/ 0, CUDNN_POOLING_MAX));
    model.add_layer(new Dense("dense1", 10));
    model.add_layer(new Softmax("softmax"));
    return model;
}