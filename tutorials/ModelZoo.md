## If you want to learn classic CNN models
### 1. 2D Convolutional Neural Network
I have builded the most popular CNN in pytorch, they are  slightly modifed to be able to accept any input image size.

You can find a list of models [```./models/__init__.py```](https://github.com/daili0015/ModelFeast/blob/master/models/__init__.py)
- [Xception](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/xception.py)
- [InceptionV3](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/inception.py)
- [InceptionResnetV2](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/inceptionresnetv2.py)
- [SqueezeNet1_0, SqueezeNet1_1](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/squeezenet.py)
- [VGG11,  VGG13, VGG16, VGG19](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/vgg.py)
- [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/resnet.py)
- [ResNext101_32x4d, ResNext101_64x4d](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/resnext.py)
- [DenseNet121, DenseNet169, DenseNet201, DenseNet161](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/densenet.py)

### 2. 3D Convolutional Neural Network

3D CNN are mainly used in video, medical image etc.

- [resnet18v2_3d, resnet34v2_3d, resnet50v2_3d, resnet101v2_3d, resnet152v2_3d, resnet200v2_3d](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/resnetv2.py)
- [resnext50_3d, resnext101_3d, resnext152_3d](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/resnext.py)
- [densenet121_3d, densenet169_3d, densenet201_3d, densenet264_3d](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/densenet.py)
- [resnet10_3d, resnet18_3d, resnet34_3d, resnet101_3d, resnet152_3d, resnet200_3d](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/resnet.py)
- [wideresnet50_3d](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/wideresnet.py)
- [i3d50, i3d101, i3d152](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/i3d.py)

I easily achieve 9th in [the medical image classification competition](https://www.datafountain.cn/competitions/335/details/weekly-rank) with DenseNet201_3d in ModelFeast!


### 3. CNN-RNN

CNN-RNN are mainly used in video, medical image etc.
This part is still on progress. Not avalible to train now, but model architecture can been seen [here](https://github.com/daili0015/ModelFeast/blob/master/models/CRNN/CRNN_module.py).
