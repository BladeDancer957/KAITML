import torch
import torch.nn as nn
from .modules import clones, LayerNorm, SublayerConnection
from .constants import print_dims

class Encoder(nn.Module):
    "Encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) #N层transformer  克隆N个transformer encoder blocks
        self.norm = LayerNorm(layer.size) #层归一化
        
    def forward(self, x, mask,pre_states,pre_mask):
        """
            x: (batch_size, src_seq_len, d_model)
            mask: (batch_size, 1, src_seq_len)
        """
        "Pass the input token ids and mask through each layer in turn"
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
            print("{0}: mask: type: {1}, shape: {2}".format(self.__class__.__name__, mask.type(), mask.shape))
        for layer in self.layers: #通过每个transformer encoder blocks
            x = layer(x, mask,pre_states,pre_mask)
        return self.norm(x) #层归一化


class EncoderLayer(nn.Module): #编码器 transformer block
    "Encoder is made up of self-attn and feed forward layers"
    def __init__(self, size, self_attn, inc_attn,feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn #多头注意力机制
        self.inc_attn = inc_attn #增量多头注意力机制
        self.feed_forward = feed_forward #全连接层
        self.sublayer = clones(SublayerConnection(size, dropout), 3) #两个shortcut
        self.size = size
        
    def forward(self, x, mask,pre_states,pre_mask):
        """norm -> self_attn -> dropout -> add -> 
        norm -> feed_forward -> dropout -> add"""
        if pre_states is None: #self-attention encoder
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            x = self.sublayer[1](x, self.feed_forward)
        else:     #incremental encoder
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            x = self.sublayer[1](x, lambda x: self.inc_attn(x, pre_states, pre_states, pre_mask))
            x = self.sublayer[2](x, self.feed_forward)

        return x
