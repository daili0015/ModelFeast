# 制作灵活的vgg网络
## 官方接口存在什么问题
在pytorch的torchvision.model模块里已经有了vgg系列网络的接口调用：

```python
from torchvision import models
net1 = models.vgg19_bn(pretrained = True)
net2 = models.vgg19_bn()
print(net2) #查看网络结构
```
这个model模块对vgg的封装[在这里](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)，可以看到他们的源码中vgg类的前向传播部分：

```python
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            #经过前面的卷积层，来到全连接之后，这里的数据尺寸已经变成了7*7，通道为512.
            #这个尺寸7*7是定死了的，这要求我数据输入必须是224*224的，很不方便。
            #这让我很不爽
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x) 
        #送到classifier这里来的数据必须是7*7*512的，如前所述
        return x
```
注意我的注释部分，在网络的全连接层，是必须把输入输出的尺寸定死下来的，这里(7, 7)是因为它使用的与训练模型就是这个尺寸的。
而vgg包含5个MaxPool层，会把输入图像缩小 ``` 2^5 ``` 倍，所以他的输入必须是 ``` 224*224 ```的。

如果我的图像长宽是512的，缩减到224可能会造成信息的丢失，影响我的准确率；或者是``` 32*32 ```的，根本不需要放大到224那么大。

所以， 我希望**能灵活地定义这个值**。
当然了 ，改变网络的尺寸后，就不能用它提供的预训练模型了。

这就是我们这篇教程的目的。

## VGG结构解析

vgg就是无脑堆加卷积层，每个几个卷积层加一个MaxPool降低尺寸，最后再末尾加一个全连接层，最终输出一个``` 1*1000 ```的向量（如果我有1000个类），每个值代表这个图被判定为一个类别的概率。
<center>
<img src="vgg16.png" width="100%" height="50%" />
Fig 1. vgg16
</center>
我们分为前面的<font color=Indigo face="黑体">卷积层部分</font>，以及后面的<font color=Indigo face="黑体">全连接</font>来构建。

### 卷积层部分
#### 固定的参数
所有的卷积核kernal都是无脑的```3*3```，padding都是```1```，stride都是```1```，根据著名的*kps公式*（我起的名字）：

```math
New_H = (H-k+2*p)/s+1 = (H-3+2*1)/1+1 = H
```
这么设定卷积层参数之后，必然不会改变图像大小；而pool都是```2*2```的，stride为```2```，padding为```0```，根据*kps公式*，pool层会把图像变为原来的```1/2```。

#### 可变的参数
由于卷积层，pool层的参数基本都定了，所以我们只需要每个卷积层输出的通道数，然后无脑堆叠就可以了。
我们无脑地用一个字典保存网络的卷积层+pool层的无脑堆叠，就像无脑的[官方实现](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)一样：
  
    
```python
vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

```
上面每一个item代表一种网络结构，数字x代表是个卷积层，其输出通道为x；'M'代表是个MaxPool层。

比如***vgg13***的参数就是```vgg_structure['B']```，如下：
```
'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

卷积层   3通道进来-64通道出去
卷积层   64通道进来-64通道出去
pool层   图像长宽减小一半
卷积层   64通道进来-128通道出去
卷积层   128通道进来-128通道出去
pool层   图像长宽减小一半
卷积层   128通道进来-256通道出去
卷积层   256通道进来-256通道出去
pool层   图像长宽减小一半
卷积层   256通道进来-512通道出去
卷积层   512通道进来-512通道出去
pool层   图像长宽减小一半
卷积层   512通道进来-512通道出去
卷积层   512通道进来-512通道出去
pool层   图像长宽减小一半
```
#### 构建卷积部分的网络
那么我们写一个函数来根据这个参数构建网络
```python
def get_Convlayer(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, 3, stride = 1, padding = 1)

# tricks： ReLU如果不inplace似乎会开辟一块新的内存，浪费空间
def construct_Conv_Block(block_cfg, img_channel):
    layer = []
    in_channel = img_channel
    for v in block_cfg:
        if v=='M':
            layer += [nn.MaxPool2d(2, stride = 2, padding = 0)]
        else:
            layer += [get_Convlayer(in_channel, v)]
            layer += [nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
            in_channel = v
    return nn.Sequential(*layer) #*代表不断去除layer中的值给函数做参数
```
如果是'M'就加一个池化层，如果是数字就加一个卷积层，最后把这个网络返回。

### 全连接层部分
这里是是问题的关键，想要实现任意尺寸输入，任意类别输出，全部在此
#### 官方实现：不灵活的全连接层
卷积部分结束，接的是全连接层，官方设计的全连接层结构是这样的
```python
def pretrained_classifier(n_class):
    return  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_class)
        )
```
这里是最关键的部分，注意他写的是```512 * 7 * 7```，说明他的全连接层的输入尺寸是完全定死的，长宽为``` 7 * 7```，512个通道。如果按照长宽为``` 7 * 7```依次向前推，我们会发现输入的图像尺寸必须是``` 224 * 224```，这很不人性化。
#### 修改全连接层：支持任意尺寸
想要让vgg适配所有大小尺寸的输入，只需要修改它的全连接层就行了；我们只要把它改成自动调整尺寸就可以了。
下面是我的实现：
```python
def adaptive_classifier(features, n_class):
    layer = []
    ratio = features//n_class 
	#从卷积最后的输出尺寸，到我们的目标：分类的类别，需要减少多少倍

    if ratio <= 256: #如果倍数不是很大，就用2个线性层得了
        h1_features = int(ratio**0.5)*n_class #hidden layer's features
        layer += [ nn.Linear(features, h1_features), nn.ReLU(True) ]
        layer += [ nn.Linear(h1_features, n_class) ]
    else: #如果倍数很大，用3个线性层，同时加入Dropout防止过拟合
        cube_root = int(ratio**0.33)
        h1_features = n_class*cube_root*cube_root
        h2_features = n_class*cube_root
        layer += [ nn.Linear(features, h1_features), nn.ReLU(True), nn.Dropout()]
        layer += [ nn.Linear(h1_features, h2_features), nn.ReLU(True), nn.Dropout() ]
        layer += [ nn.Linear(h2_features, n_class) ]

    return nn.Sequential(*layer)
```
很明显，这样的函数，会根据你设定的图像输入，自动的帮你创建合适的全连接层。

## 着手构建网络
### 构建简单的VGG类
写好了卷积层和全连接层的构建函数，我们就可以写一个VGG类了，以后我们想要使用vgg，只要用这个类创建一个vgg就可以了。
实现一个网络类非常简单，只要记住3点：
1. 定义时继承```nn.Module```这个类
2. 在```__init__```函数里创建网络的各个部分，保存为它的成员变量比如 ```self.classifier = pretrained_classifier(1000)```
3. 在```forward```函数里写清楚网络的output是怎么出来的

```python
class vgg_Net(nn.Module):
    def __init__(self, cfg):
        super(vgg_Net, self).__init__()

        # features 的命名是因为要与预训练模型的命名对应，方便加载进来
        self.features = construct_Conv_Block(cfg, 3)
        self.classifier = pretrained_classifier(1000)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
```

上面这段代码就是vgg类了，以后我们只要```model = vgg_Net(Net_cfg, param)  ```，就可以创建一个vgg网络了。

这里为了方便阅读，写的是一个简单的官方结构的类，下面我们考虑怎么把图像的尺寸送给网络，怎么调整全连接层。
### 智能调节网络的全连接层

我们前面实现了```adaptive_classifier ```这个智能调节的函数，现在我们把在vgg类的成员函数中调用它：

```python
    def adjust_classifier(self, n_class):
        self.classifier = adaptive_classifier(self.fc_in_features, n_class)
        init_weight(self.classifier)
        print("网络的线性层自动调整设置为：")
        print(self.classifier)
```
相应的vgg类做一些改变，在init初始化时，把图像长宽，卷积部分的输出保存下来：

```python
class vgg_Net(nn.Module):

    def __init__(self, cfg, param):
        '''
        param为字典，包含网络需要的参数
        param['img_height']: image's height, must be 32's multiple
        param['img_width']: image's weight, must be 32's multiple
        '''
        super(vgg_Net, self).__init__()
        self.features = construct_Conv_Block(cfg, 3)

        conv_height, conv_width = param['img_height']//32, param['img_width']//32
        self.fc_in_features = conv_height * conv_width * 512 #全连接层的输入通道

        self.classifier = pretrained_classifier(1000)

```
这样就可以了，在载入模型的时候，我们首先
1. 创建一个跟官方版本一样结构的VGG（224*224 输出；类别为1000）
2. 把官方的预训练模型导入进来，权值复制给我们的VGG
3. 修改我们的vgg的全连接层部分，丢掉原来的官方的那些全连接层，换成我们自己的

这3步环环相扣，依次执行；只有在1中创建了跟官方版本一样结构的VGG，我们才能把官方的预训练模型权重赋值给vgg，不然结构都不一样根本没法赋值；赋值完了再修改最后的全连接层。
这些全部写在函数里，就是：
```python
def get_vgg(Net_cfg, Net_urls, file_name, n_class, pretrained=False,
            img_size=(224, 224)):
    '''
    Net_cfg：网络结构
    Net_urls：预训练模型的url
    file_name：预训练模型的名字
    n_class：输出类别
    pretrained：是否使用预训练模型

    '''
    if isinstance(img_size, tuple):
        img_height, img_width = img_size[0], img_size[1]
    else:
        img_height = img_width = img_size

    param = {'img_height':img_height, 'img_width':img_width}

    model = vgg_Net(Net_cfg, param) #先建立一个跟预训练模型一样的网络
    
    if pretrained: #把权值导入进来，通过本地文件或者从网络下载
        if os.path.exists(os.path.join("./pretrained", file_name)):
            model.load_state_dict(TorchLoad(os.path.join("./pretrained", file_name)))
            logging.info("Find local model file, load model from local !!")
            logging.info("找到本地下载的预训练模型！！直接载入！！")
        else:
            logging.info("pretrained 文件夹下没有，从网上下载 !!")
            model.load_state_dict(model_zoo.load_url(Net_urls, model_dir = "./pretrained/"))
            logging.info("下载完毕！！载入权重！！")

    model.adjust_classifier(n_class) #调整全连接层，迁移学习

    return model

```
这样，我们就实现了一个允许输入任何尺寸图像，输出任意分类数的vgg网络，这可比官方的只允许```224*224```，只能输出```1000```类可舒服多了！

还剩下一些封装的代码，都在下面两个文件里：
1. vgg_Net.py 里面是vgg类的定义
2. vgg.py是用来封装vgg类的，提供调用的接口，包括不同vgg的结构配置，预训练模型下载地址等等

最终，我们只需要一行语句，就可以创建任意尺寸输入，任意类别输出的vgg来训练了！
```python
model = vgg16(10, 32, True)
```
10是分类数，32是图像尺寸大小（如果你的图像不是正方形，你可以直接把32换成（高，宽），非常方便 ），最后指明是否使用预训练的权重。
