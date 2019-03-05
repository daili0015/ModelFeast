## If you want to learn classic CNN models
### 1. 2D Convolutional Neural Network
I have builded the most popular CNN in pytorch, they are  slightly modifed to be able to accept any input image size.

You can find them under ```./models/classifiers```
- Xception
- InceptionV3
- InceptionResnetV2
- SqueezeNet1_0, SqueezeNet1_1
- VGG11,  VGG13, VGG16, VGG19
- ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- ResNext101_32x4d, ResNext101_64x4d
- DenseNet121, DenseNet169, DenseNet201, DenseNet161

### 2. 3D Convolutional Neural Network

3D CNN are mainly used in video, medical image etc.

- resnet18v2_3d, resnet34v2_3d, resnet50v2_3d, resnet101v2_3d, resnet152v2_3d, resnet200v2_3d
- resnext50_3d, resnext101_3d, resnext152_3d
- densenet121_3d, densenet169_3d, densenet201_3d, densenet264_3d
- resnet10_3d, resnet18_3d, resnet34_3d, resnet101_3d, resnet152_3d, resnet200_3d
- wideresnet50_3d
- i3d50, i3d101, i3d152

I easily achieve 9th in [the medical image classification competition](https://www.datafountain.cn/competitions/335/details/weekly-rank) with DenseNet201_3d in ModelFeast!


### 3. CNN-RNN

CNN-RNN are mainly used in video, medical image etc.
This part is still on progress. Not avalible to train now, but model architecture can been seen [here](https://www.datafountain.cn/competitions/335/details/weekly-rank) .
