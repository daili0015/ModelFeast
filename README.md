# ModelFeast
ModelFeast is more than model zoo!
It is:
- [A pytorch project template](https://github.com/daili0015/ModelFeast#if-you-want-to-use-the-pytorch-project-template)
- [A gather of the most popular 2D, 3D CNN models](https://github.com/daili0015/ModelFeast#if-you-want-to-learn-classic-cnn-models)
- [A Scaffold to make deep learn much more simply and flexibly](https://github.com/daili0015/ModelFeast#if-you-want-to-train-model)

The features are more you could think of:
- Load data, initialize, train and save model within 3 lines !
- Include the most popular 2D CNN, 3D CNN, and CRNN models !
-  Allow any input image size (pytorch official model zoo limit your input size harshly) !
- Help you sweep all kinds of classification competitions (convenient api for ensemble learning).

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


## If you want to train model
### 1. Prepare your dataset

For example, you can put image dataset in a folder.
I put it here: ```E:/Oxford_Flowers17/train```
<center>
<img src="tutorials/datase_path.png" width="40%" height="20%" />
</center>
Here are some datasets available:

| Baidu Yun | Google |
| :------: | :------: |
| [flowers17](https://pan.baidu.com/s/16PjFHGJf-IRxlIdxBz2LYQ)(kkdc) | [flowers17](https://drive.google.com/open?id=11h4O0V-qZ2OwEVd_MxeETN0AQLDvRtRL) |
| [SceneClass13](https://pan.baidu.com/s/1yLTLtVBgmHRPOZN65pnGrw)(0onp) | [SceneClass13](https://drive.google.com/open?id=1wxlpGjY9eKMrgn5FjXQVc5oN_6CP4TF9) |
| [AnimTransDistr](https://pan.baidu.com/s/1cDdfb8vJnZTPt-w3lMMulQ)(otd5) | [AnimTransDistr](https://pan.baidu.com/s/16PjFHGJf-IRxlIdxBz2LYQ) |

ps: Of course you can define dataloader on your own!
### 2. Train  within 3 lines
#### You can train from zero
```python
if __name__ == '__main__': # removing this line brings dataloader error, this is because of python's multithread feature
    clf = classifier('xception', 17, (200, 200), 'E:/Oxford_Flowers17/train')
    clf.train()
```
It will begin to train a Xception with dataloader with 17 classes, resize image to 200*200, load data from ```'E:/Oxford_Flowers17/train'```. Best model will be saved to folder ```"./saved"```every 2 epoch. To know more about default setting, click [here](https://pan.baidu.com/s/1cDdfb8vJnZTPt-w3lMMulQ).
#### Or resume previous training
```python
if __name__ == '__main__':
    clf = classifier('xception', 17, (60, 60), 'E:/Oxford_Flowers17/train')
    clf.train_from('E:/ModelFeast/saved/xception/0305_130143/checkpoint_best.pth')
```
unbelievably simple, right ?!

## If you want to use the pytorch project template
### 1. Define your own dataloader
Define your own dataloader, please inherit ```BaseDataLoader``` in ```./base/base_data_loader.py``` (instead of the official version ```torch.utils.data.DataLoader```), this will save you much time and workload.
Having trouble with how to define ?  Read ```./data_loader/data_loaders.py```may give you some hint! 
### 2. Define your loss and metric
For classification, using ```cls_loss``` in ```./models/loss.py``` and ```accuracy``` in ```./models/metric.py``` is just fine.
### 3. Define your model
There are a lot of models you can use dircetly, or you can define it youself.
### 4. Train it
It's recommended to use the ```classifier``` in ```classifier.py```, it's really convenient to train, resume , and very flexible to change default settings (learning rate cheduler, optimizer, epoch...)

## Reference
[https://github.com/lanpa/tensorboardX](https://github.com/lanpa/tensorboardX)
[https://github.com/pytorch/vision/tree/master/torchvision/models](https://github.com/pytorch/vision/tree/master/torchvision/models)
[https://github.com/kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
[https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)
[https://github.com/AlexHex7/Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch)
[https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)