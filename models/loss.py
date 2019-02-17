import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

import torch
w = torch.Tensor([1.5, 2.1])
w = w.cuda()
def cls_loss(output, target):
    return F.cross_entropy(output, target, weight=w)


if __name__ == '__main__':
    from types import FunctionType
    cls = cls_loss1()
    print(type(cls))
    print(type(cls_loss))
    print( isinstance(cls_loss, FunctionType) )