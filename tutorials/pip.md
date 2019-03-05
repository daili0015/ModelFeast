## try modelfeast via pip
### 1. install
```python
pip3 install modelfeast
```
### 2. get a model
```python
from modelfeast import *
model = squeezenet(n_class=10, img_size=(224, 224), pretrained=True)
print(model)
```
The interface to create a 2D CNN model can be used in the manner:
```python
model = modelname(n_class=10, img_size=256, pretrained=True, pretrained_path="./pretrained/")
```
Check [```./models/__init__.py```](https://github.com/daili0015/ModelFeast/blob/master/models/__init__.py) to see avaliable modelname.


### 3. train a model using modelfeast
```python
from modelfeast import *
if __name__ == '__main__':
    clf = classifier('xception', 17, (30, 30), 'E:/Oxford_Flowers17/train')
    clf.train()
```
The class ```classifier``` is very flexible.

You can define a model on your own, and train it using ```classifier```.
```python
from modelfeast import *
from torch import nn

#define your own model
class FuckerNet(nn.Module):

    def __init__(self):
        super(dal_BN, self).__init__()
        self.sq1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding = 1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5), #padding = 0 , stride=1,默认
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )
        self.linear = nn.Linear(400 ,10) 

    def forward(self, x):
        x = self.sq1(x)
        y = self.linear(x.view(x.shape[0], -1))
        return y


if __name__ == '__main__':

	model = FuckerNet()
    clf = classifier(model=model, 17, (30, 30), 'E:/Oxford_Flowers17/train')
    clf.train()

```

You can define your own dataloader, optimizer, lr_schedule, loss, metric and use ```classifier``` to do the rest !
To learn more, please read classifier.py.

Life is short, there's no reason to spend time on meaningless things. So, enjoy modelfeast !