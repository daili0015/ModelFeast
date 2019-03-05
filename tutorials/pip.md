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
model = modelname(n_class=10, img_size=(224, 224), pretrained=True, pretrained_path="./pretrained/")
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

class FireNet(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim):
		super(dai_Net, self).__init__()
		self.layer1 = nn.Conv2d(3, 8)
		self.layer2 = nn.Tanh()
		self.layer3 = nn.Linear(8, 16)
		self.layer4 = nn.Sigmoid()
	def forward(self, x):
		fx = self.layer1(x)
		fx = self.layer2(fx)
		fx = self.layer3(fx)
		fx = self.layer4(fx)
		return fx

if __name__ == '__main__':

	model = FireNet()
    clf = classifier(model=model, 17, (30, 30), 'E:/Oxford_Flowers17/train')
    clf.train()

```

You can define your own dataloader, optimizer, lr_schedule, loss, metric and use ```classifier``` to do the rest !
To learn more, please read classifier.py.

Life is short, there's no reason to spend time on meaningless things. So, enjoy modelfeast !