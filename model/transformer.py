import copy
import torch
import torch.nn as nn

from .embedding import Embeddings, PositionalEncoding
from .attention import MultiHeadAttention, GraphAttention
from .modules import PositionWiseFeedForward
from .encoder import Encoder, EncoderLayer

from .generator import Generator
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture
    """
    def __init__(self, encoder, src_embed,  generator, context_length=6, graph_attention=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder

        self.src_embed = src_embed

        self.generator = generator
        self.graph_attention = graph_attention

        self.context_length = context_length
        
        self.embed_size = self.src_embed[0].embedding.weight.shape[1]
        self.src_mlp = nn.Linear(2*self.embed_size, self.embed_size)

        
    def forward(self, src, src_mask):
        # src: (batch_size, (context_length+1) * seq_len)
        # src_mask: (batch_size, 1, (context_length+1) * seq_len)

        # concept: (batch_size, (context_length + 1) * 30)

        #得到context 和 最后一个子句的词嵌入（添加位置编码）
        src_embed = self.src_embed(src) # src_embed: (batch_size, (context_length+1) * seq_len, d_model)

        if self.graph_attention is not None: #使用外部知识
            # graph attention to compute concept representations 计算concept 表示
            src_concept_embed = self.graph_attention(src, src_embed) #(batch_size, (context_length+1) * seq_len, d_model)

            # concatenate concept representations with utterance embeddings 和词嵌入进行拼接
            src_embed = self.src_mlp(torch.cat([src_embed, src_concept_embed], dim=-1)) ##(batch_size, (context_length+1) * seq_len, d_model)

        
        context_length = self.context_length + 1 #context + target utterance
        seq_len = src.shape[1]//context_length

        pre_states = None
        pre_mask = None

        for i in range(context_length):
            s, e = i*seq_len, (i+1)*seq_len
            x =self.encoder(src_embed[:,s:e,:], src_mask[:,:,s:e],pre_states,pre_mask)
            pre_states = x
            pre_mask = src_mask[:,:,s:e]

        x = F.max_pool1d(x.permute(0, 2, 1), x.shape[1]).squeeze(-1)  # (batch_size, d_model)

        return x


def make_model(src_vocab, N=6, d_model=512, d_ff=2048, h=8, output_size=0, dropout=0.1, KB=False, \
         context_length=0):

    c = copy.deepcopy
    #编码器多头注意力机制
    enc_attn = MultiHeadAttention(h, d_model)

    #编码器增量多头注意力机制
    enc_inc = MultiHeadAttention(h, d_model)
    #全连接层  d_model->d_ff->d_model
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    #位置编码
    position = PositionalEncoding(d_model, dropout)
    #词嵌入层
    emb = Embeddings(d_model, src_vocab)
    concept_embed = None
    graph_attention = None
    if KB: #如果使用外部知识
        #图注意力机制
        graph_attention = GraphAttention(vocab_size=src_vocab, embed_size=d_model, context_length=context_length, concept_embed=concept_embed)
    
    #构建模型对象
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(enc_attn),c(enc_inc), c(ff), dropout), N),
        nn.Sequential(emb, c(position)),
        Generator(d_model, output_size),
        context_length,
        graph_attention=graph_attention
    )

    return model






