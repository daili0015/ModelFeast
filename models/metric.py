import torch
import numpy as np

def topK_accuracy(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# 与top1_acc是同一个东西，就是分类准确率；不过是用了numpy
def accuracy(output, target):
    with torch.no_grad():
        y_pred = output.cpu().max(1)[1]  #返回（最大值，最大值下标）;与np不同
        acc = np.array((y_pred.data-target.cpu().data)==0).mean() #变成np，可以直接np.array()
    return acc

def top1_acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)