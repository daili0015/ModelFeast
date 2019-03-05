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
