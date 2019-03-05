# ModelFeast
ModelFeast is more than model zoo!
It is:
- A pytorch project template.
- A gather of the most popular models.
- A Scaffold to make deep learn much more simply and flexibly.

The features are more you could think of:
- Detailed tutorials !
- Load data, initialize, train and save model within 3 lines !
- Include the most popular 2D CNN, 3D CNN, and CRNN models !
-  Allow any input image size (pytorch official model zoo limit your input size harshly) !
- Help you sweep all kinds of classification competitions (convenient api for ensemble learning).

## Get started
### 1. Prepare your dataset

For example, you can put image dataset in a folder.
I put it here: ```E:/Oxford_Flowers17/train```
<center>
<img src="tutorials/datase_path.png" width="90%" height="40%" />
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
if __name__ == '__main__': # removing this line brings dataloader error, this is because of python's multithread feature
    clf = classifier('xception', 17, (200, 200), 'E:/Oxford_Flowers17/train')
    clf.train()
```
