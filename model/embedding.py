import math

import torch
import torch.nn as nn

from .constants import print_dims

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model) #词嵌入矩阵
        self.d_model = d_model #embedding_size
        
    def forward(self, x):
        """
            x: (batch_size, seq_len)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return self.embedding(x) * math.sqrt(self.d_model) #把词索引映射为词向量 再乘以embedding_size的开方 (使用Xavier初始化方式，使方差为1)

class PositionalEncoding(nn.Module):
    #位置编码
    #sin cos产生的绝对位置编码

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) #丢弃率
        
        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        x = x + self.pe[:, :x.size(1)] #词嵌入+位置编码
        return self.dropout(x) #通过dropout
