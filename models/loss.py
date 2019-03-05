import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cls_loss(output, target):
    return F.cross_entropy(output, target)  #, weight=w

def mse_loss(output, target):
    target = target.float().unsqueeze(1)
    return F.binary_cross_entropy_with_logits(output, target, pos_weight=w)  #, weight=w


if __name__ == '__main__':
    from types import FunctionType
    cls = cls_loss1()
    print(type(cls))
    print(type(cls_loss))
    print( isinstance(cls_loss, FunctionType) )