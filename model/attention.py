import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import clones
from .constants import print_dims

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot product self attention
        query: (batch_size, h, seq_len, d_k), seq_len can be either src_seq_len or tgt_seq_len
        key: (batch_size, h, seq_len, d_k), seq_len in key, value and mask are the same
        value: (batch_size, h, seq_len, d_k)
        mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, tgt_seq_len, tgt_seq_len) (legacy)
    """
    if print_dims:
        print("{0}: query: type: {1}, shape: {2}".format("attention func", query.type(), query.shape))
        print("{0}: key: type: {1}, shape: {2}".format("attention func", key.type(), key.shape))
        print("{0}: value: type: {1}, shape: {2}".format("attention func", value.type(), value.shape))
        print("{0}: mask: type: {1}, shape: {2}".format("attention func", mask.type(), mask.shape))
    d_k = query.size(-1)

    # scores: (batch_size, h, seq_len, seq_len) for self_attn, (batch_size, h, tgt_seq_len, src_seq_len) for src_attn
    scores = torch.matmul(query, key.transpose(-2, -1)/math.sqrt(d_k))  #计算注意力得分
    # print(query.shape, key.shape, mask.shape, scores.shape)

    if mask is not None:  #填充部分 对应的注意力得分 置为很小的数
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1) #softmax 转换为权重
    if dropout is not None:
        p_attn = dropout(p_attn) #通过dropout
    return torch.matmul(p_attn, value), p_attn #返回多头注意力机制的计算结果

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0 #embedding_size 要整除注意力头数
        self.d_k = d_model//h #划分为多个子空间 多头
        self.h = h #头数
        self.linears = clones(nn.Linear(d_model, d_model), 4) #用于产生Q，K，V的Dense层
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parameter(torch.Tensor([1]))
        
    def forward(self, query, key, value, mask=None):
        """
            query: (batch_size, seq_len, d_model), seq_len can be either src_seq_len or tgt_seq_len
            key: (batch_size, seq_len, d_model), seq_len in key, value and mask are the same
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len) or (batch_size, tgt_seq_len, tgt_seq_len) (legacy)
        """
        if print_dims:
            print("{0}: query: type: {1}, shape: {2}".format(self.__class__.__name__, query.type(), query.shape))
            print("{0}: key: type: {1}, shape: {2}".format(self.__class__.__name__, key.type(), key.shape))
            print("{0}: value: type: {1}, shape: {2}".format(self.__class__.__name__, value.type(), value.shape))
            print("{0}: mask: type: {1}, shape: {2}".format(self.__class__.__name__, mask.type(), mask.shape))
        if mask is not None: #mask非空
            mask = mask.unsqueeze(1) #添加一个0轴
        nbatches = query.size(0) #batch 大小
        
        # 1) Do all linear projections in batch from d_model to (h, d_k)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) #（batch_size,h,seq_len,d_k）
            for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch 并行计算多头注意力
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout) # (batch_size, h, seq_len, d_k)
        if print_dims:
            print("{0}: x (after attention): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))

        # 3) Concatenate and apply a final linear 多头结果拼接 再通过一个全连接层
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) #(batch_size,seq_len,h*d_k=d_model)
        x = self.linears[-1](x) # (batch_size, seq_len, d_model)
        if print_dims:
            print("{0}: x (after concatenation and linear): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return x


class GraphAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, context_length, concept_embed=None):
        super(GraphAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        if concept_embed is None: #concept嵌入层
            self.concept_embed = nn.Embedding(vocab_size, embed_size)#初始化concept嵌入矩阵 
        else:
            self.concept_embed = concept_embed

        self.context_length = context_length

    def init_params(self,edge_matrix=None,relation_matirx=None):
        edge_matrix_range = (edge_matrix.max(dim=1)[0] - edge_matrix.min(dim=0)[0]).unsqueeze(1)
        edge_matrix = edge_matrix/(edge_matrix_range + (edge_matrix_range==0).float()) # normalization

        self.relation_matrix = relation_matirx
        self.edge_matrix = edge_matrix

    def forward(self, src, src_embed): #双层图注意力机制
        # src: (batch_size, (context_length+1) * seq_len)
        # src_embed: (batch_size, (context_length+1)*seq_len, d_model)

        # embed: shared embedding layer: (vocab_size, d_model)

        scores = []
        base = torch.matmul(src_embed, self.concept_embed.weight.transpose(0,1))
        device = src.device
        self.relation_matrix = self.relation_matrix.to(device)
        self.edge_matrix = self.edge_matrix.to(device)
        concept_embeddings=[]
        for i in range(8):
            edge_matrix = (self.relation_matrix == (i + 1)).float() * self.edge_matrix
            concept_weights = (edge_matrix[src] > 0).float() * base  # (batch_size, (context_length+1)*seq_len, vocab_size)
            del edge_matrix
            concept_embedding = torch.matmul(torch.softmax(concept_weights, dim=2),self.concept_embed.weight)
            del concept_weights
            concept_embeddings.append(concept_embedding)
            scores.append((src_embed * concept_embedding).sum(dim=2, keepdim=True))
            del concept_embedding
            

        scores = torch.cat(scores, dim=2)
        weights = torch.softmax(scores, dim=2)
        del scores

        final_embedding = torch.zeros_like(src_embed)
        for i in range(8):
            final_embedding += weights[:,:,i].unsqueeze(2)*concept_embeddings[i]
        
        return final_embedding












    

