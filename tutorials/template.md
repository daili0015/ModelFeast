## If you want to use the pytorch project template
### 1. Define your own dataloader
Define your own dataloader, please inherit [```BaseDataLoader```](https://github.com/daili0015/ModelFeast/blob/master/base/base_data_loader.py#L7) in [```./base/base_data_loader.py```](https://github.com/daili0015/ModelFeast/blob/master/base/base_data_loader.py) (instead of the official version ```torch.utils.data.DataLoader```), this will save you much time and workload.

Having trouble with how to define ?  Read [```./data_loader/data_loaders.py```](https://github.com/daili0015/ModelFeast/blob/master/data_loader/data_loaders.py)may give you some hint! 
### 2. Define your loss and metric
For classification, using [```cls_loss```](https://github.com/daili0015/ModelFeast/blob/master/models/loss.py#L7) in [```./models/loss.py```](https://github.com/daili0015/ModelFeast/blob/master/models/loss.py) and [```accuracy```](https://github.com/daili0015/ModelFeast/blob/master/models/metric.py#L14) in [```./models/metric.py```](https://github.com/daili0015/ModelFeast/blob/master/models/metric.py) is just fine.
### 3. Define your model
There are a lot of models you can use dircetly, or you can define it youself.
### 4. Train it
It's recommended to use the [```classifier```](https://github.com/daili0015/ModelFeast/blob/master/classifier.py#L24) in [```classifier.py```](https://github.com/daili0015/ModelFeast/blob/master/classifier.py), it's really convenient to train, resume , and very flexible to change default settings (learning rate cheduler, optimizer, epoch...)
