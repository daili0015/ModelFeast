# ModelFeast
[中文版readme](https://github.com/daili0015/ModelFeast/blob/master/README_cn.md)

Please star ModelFeast if it helps you. This is very important to me! Thanks very much !

ModelFeast is more than model-zoo!
It is:
- [A gather of the most popular 2D, 3D CNN models](https://github.com/daili0015/ModelFeast/blob/master/tutorials/ModelZoo.md)
- [A tool to make deep learn much more simply and flexibly](https://github.com/daili0015/ModelFeast/blob/master/tutorials/Scaffold.md)
- [A pytorch project template](https://github.com/daili0015/ModelFeast/blob/master/tutorials/template.md)

[What is ModelFeast ?](https://github.com/daili0015/ModelFeast/blob/master/tutorials/what'sit.md)


## Avalible models
### 2D CNN
- [Xception](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/xception.py)
- [InceptionV3](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/inception.py)
- [InceptionResnetV2](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/inceptionresnetv2.py)
- [SqueezeNet1_0, SqueezeNet1_1](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/squeezenet.py)
- [VGG11,  VGG13, VGG16, VGG19](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/vgg.py)
- [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/resnet.py)
- [ResNext101_32x4d, ResNext101_64x4d](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/resnext.py)
- [DenseNet121, DenseNet169, DenseNet201, DenseNet161](https://github.com/daili0015/ModelFeast/blob/master/models/classifiers/densenet.py)

Pretrained models ( trained on ImageNet ) for 2D CNN is now avalible on [Baiduyun](https://pan.baidu.com/s/1bfZj7gxyFSiKHf6cYEeLwA)(fst6) and [Google Drive](https://drive.google.com/open?id=1BUGf-l6IMHaZQ9LFGUSDsl_MzDtT91nT)

### 3D CNN
- [resnet18v2_3d, resnet34v2_3d, resnet50v2_3d, resnet101v2_3d, resnet152v2_3d, resnet200v2_3d](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/resnetv2.py)
- [resnext50_3d, resnext101_3d, resnext152_3d](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/resnext.py)
- [densenet121_3d, densenet169_3d, densenet201_3d, densenet264_3d](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/densenet.py)
- [resnet10_3d, resnet18_3d, resnet34_3d, resnet101_3d, resnet152_3d, resnet200_3d](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/resnet.py)
- [wideresnet50_3d](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/wideresnet.py)
- [i3d50, i3d101, i3d152](https://github.com/daili0015/ModelFeast/blob/master/models/StereoCNN/i3d.py)

### CNN-RNN
This part is still on progress. Not avalible to train now, but model architecture can been seen [here](https://github.com/daili0015/ModelFeast/blob/master/models/CRNN/CRNN_module.py).

## Get started
Determine what you need and read corresponding tutorials
- [I want to train a model as simple as possible](https://github.com/daili0015/ModelFeast/blob/master/tutorials/Scaffold.md)
- [I just need the codes of CNNs ](https://github.com/daili0015/ModelFeast/blob/master/tutorials/ModelZoo.md)
- [I need a standard pytorch project template](https://github.com/daili0015/ModelFeast/blob/master/tutorials/template.md)

Or you can use modelfeast simply via pip !
```
pip3 install modelfeast
```
[pip user guide](https://github.com/daili0015/ModelFeast/blob/master/tutorials/pip.md)

## Features
The features are more than you could think of:
- Train and save model within 3 lines !
- Multi GPU support !
- Include the most popular 2D CNN, 3D CNN, and CRNN models !
-  Allow any input image size (pytorch official model zoo limit your input size harshly) !
- Help you sweep all kinds of [classification competitions](https://github.com/daili0015/ModelFeast/blob/master/tutorials/ModelZoo.md#2-3d-convolutional-neural-network).

## Reference
[https://github.com/lanpa/tensorboardX](https://github.com/lanpa/tensorboardX)

[https://github.com/pytorch/vision/tree/master/torchvision/models](https://github.com/pytorch/vision/tree/master/torchvision/models)

[https://github.com/kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)

[https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)

[https://github.com/AlexHex7/Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch)

[https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

