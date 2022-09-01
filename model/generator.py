import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import print_dims

class Generator(nn.Module): #输出层 做分类
    """
    多任务学习，proj1 原始情感分类任务，多分类；proj2 neutral vs. non-neutral 2分类；proj3 把non-neutral细分为具体的情感类别 做一个(向量)回归
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj1 = nn.Linear(d_model, vocab)
        self.proj2 = nn.Linear(d_model, 2)
        self.proj3 = nn.Linear(d_model, vocab-1)
        
    def forward(self, x):
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        o1 = self.proj1(x)
        o2 = self.proj2(x)
        o3 = F.sigmoid(self.proj3(x))#sigmoid将输出归一化到0-1之间
        return o1,o2,o3  
