import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import print_dims

class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, n_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(n_features))
        self.b_2 = nn.Parameter(torch.zeros(n_features))
        self.eps = eps
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean)/(std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    The norm is applied first as apposed to last
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
            x: (batch_size, seq_len, d_model)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))

class PositionWiseFeedForward(nn.Module):
    #全连接层

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff) #第一个全连接层
        self.w2 = nn.Linear(d_ff, d_model) #第二个全连接层
        self.dropout = nn.Dropout(dropout) #丢弃率
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        #d_model->d_ff->d_model  第一个全连接层使用relu激活函数 第二个没有激活函数
        return self.w2(self.dropout(F.relu(self.w1(x))))


def clones(module, N): #克隆N个相同的module
    """
    Produce N identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

