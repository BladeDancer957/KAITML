import os
import random
import logging
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn.init import xavier_uniform_
from pymagnitude import Magnitude

from utils.io import load_pickle,to_pickle
from utils.data import Vocab, convert_examples_to_ids, create_batches, merge_splits, get_vocab_embedding, \
    filter_conceptnet, remove_KB_duplicates
from utils.tools import count_parameters, label_distribution_transformer

from model.transformer import make_model
from model.generator import Generator
from model.batch import flatten_examples_classification, create_batches_classification
from model.loss import SimpleLossCompute

from torch.optim.lr_scheduler import ReduceLROnPlateau
logging.basicConfig(level=logging.INFO, \
                    format='%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt="%Y-%m-%d-%H-%M-%S")

if __name__ == "__main__":

    # 声明argparse对象
    parser = argparse.ArgumentParser(description="Model for Context-based Emotion Classification in Conversations")

    # 数据集
    parser.add_argument('--dataset', type=str, required=True)
    # 最小词频
    parser.add_argument('--min_freq', type=int, default=1)
    # 最大 词典大小
    parser.add_argument('--max_vocab_size', type=int, default=1e9)
    # 上下文窗口大小 M
    parser.add_argument('--context_length', type=int, default=6)
    # 测试模式 action="store_true" 出现该参数 等价于 --test_mode=True
    parser.add_argument('--test_mode', action="store_true")
    # 知识库
    parser.add_argument('--KB', type=str, default="conceptnet")
    # 采用知识库的比例
    parser.add_argument('--KB_percentage', type=float, default=1.0)
    # 层数 transformer bolck数
    parser.add_argument('--n_layers', type=int, default=1)  # 1 layer in paper
    # embedding size
    parser.add_argument('--d_model', type=int, default=100)
    #全联接层维度
    parser.add_argument('--d_ff', type=int, default=100)
    # 多头注意力 头数
    parser.add_argument('--h', type=int, default=4)

    # 训练epoch数
    parser.add_argument('--epochs', type=int, default=10)
    # batch大小
    parser.add_argument('--batch_size', type=int, default=64)
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-4)
    # 丢弃率
    parser.add_argument('--dropout', type=float, default=0)
    # 随机种子
    parser.add_argument('--seed', type=int, default=1)
    #学习率衰减
    parser.add_argument('--decay',type=bool,default=False)
    #两个辅助任务的权重
    parser.add_argument('--lambda1',type=float, default=0)
    parser.add_argument('--lambda2', type=float, default=0)
    # 解析参数
    args = parser.parse_args()

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()  # gpu数量
    print("training on {0} gpus!".format(args.n_gpu))

    # 把命令行参数 赋给相应全局变量
    test_mode = args.test_mode
    dataset = args.dataset
    min_freq = args.min_freq
    max_vocab_size = int(args.max_vocab_size)

    KB = args.KB
    KB_percentage = args.KB_percentage
    context_length = args.context_length
    n_layers = args.n_layers
    d_model = args.d_model
    d_ff = args.d_ff
    h = args.h
    
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    embedding_size = d_model

    if dataset == "EC":  # EC数据集的上下文窗口为2
        context_length = 2

    # 训练相关参数
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout
    seed = args.seed

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    # 加载相应数据集的样本 训练集和验证集
    logging.info("Loading data...")
    train = load_pickle("./data/{0}/train.pkl".format(dataset))
    val = load_pickle("./data/{0}/val.pkl".format(dataset))

    if test_mode: 
        print("mode test.")
        test = load_pickle("./data/{0}/test.pkl".format(dataset))
        train = merge_splits(train, val)
        val = test

    logging.info("Number of training examples: {0}".format(len(train)))
    logging.info("Number of validation examples: {0}".format(len(val)))

    for ex in train[0][:3]:
        logging.info("Examples: {0}".format(ex))

    logging.info("Building vocab...")
    # 构建词典
    vocab = Vocab(train, min_freq, max_vocab_size)
    vocab_size = len(vocab.word2id)  # 词典大小
    logging.info("Vocab size: {0}".format(vocab_size))
    
    if dataset=="IEMOCAP": #让IEMOCAP数据集的neutral类/标签对应0，与其他数据集一致。
        vocab.emotion2id[5] = 1
        vocab.emotion2id[2] = 0

    # build vocab and data
    # use pretrained word embedding

    logging.info("Loading word embedding from Magnitude...")
    #通过Magnitude工具加载预训练 词向量
    home = os.path.expanduser("~") #需要把预训练词向量文件 转换为 .magnitude格式 放在home目录的WordEmbedding路径下
    if embedding_size in [50, 100, 200]:
        #vectors = Magnitude("glove/medium/glove.twitter.27B.{0}d.magnitude".format(embedding_size))
        vectors = Magnitude(os.path.join(home, "WordEmbedding/glove.twitter.27B.{0}d.magnitude".format(embedding_size)))
    elif embedding_size in [300]:
        vectors = Magnitude(os.path.join(home, "WordEmbedding/glove.840B.{0}d.magnitude".format(embedding_size)))

    #获取词嵌入矩阵（用预训练词向量初始化的）
    pretrained_word_embedding = get_vocab_embedding(vocab, vectors, embedding_size)

    if KB == "conceptnet":
        path = '5.6.1'
        # 计算边矩阵
        # 加载所选数据集 对应的处理好的知识库.pkl文件
        conceptnet = load_pickle("./data/KB/{0}/{1}.pkl".format(path,dataset))
        # 把不在词典（去除了所有子句中的低频词）中的concept 从conceptnet（所有子句的单词/1-gram）中去掉
        # 并且把conceptnet 置信度<=1的concept去掉
        filtered_conceptnet = filter_conceptnet(conceptnet, vocab)
        # 去重
        filtered_conceptnet = remove_KB_duplicates(filtered_conceptnet)
        vocab_size = len(vocab.word2id)  # 词典大小
        # 初始化边矩阵为0
        edge_matrix = np.zeros((vocab_size, vocab_size))
        #初始化关系矩阵为0
        relation_matirx = np.zeros((vocab_size, vocab_size))

        # 边矩阵过滤后的conceptnet对应的邻接矩阵 节点为每个concept，值为两个concept之间的置信度（concept全都包含在对应数据集的词典中）
        #关系矩阵节点为每个concept，值为两个concept之间的关系代号（1-38  38种关系）
        #我们选择出现频率较高且与语义密切相关的6个关系单独处理，其余频率比较低的关系统一作为第7种关系。节点与自身的连接作为第8种关系。处理后总共8种关系。
        for k in filtered_conceptnet:
            for c, w, r in filtered_conceptnet[k]:
                if r=="RelatedTo":
                    relation_matirx[vocab.word2id[k], vocab.word2id[c]] = 1
                elif r=="Synonym":
                    relation_matirx[vocab.word2id[k], vocab.word2id[c]] = 2
                elif r=="HasContext":
                    relation_matirx[vocab.word2id[k], vocab.word2id[c]] = 3
                elif r=="IsA":
                    relation_matirx[vocab.word2id[k], vocab.word2id[c]] = 4
                elif r=="SimilarTo":
                    relation_matirx[vocab.word2id[k], vocab.word2id[c]] = 5
                elif r=="Antonym": 
                    relation_matirx[vocab.word2id[k], vocab.word2id[c]] = 6
                else:
                    relation_matirx[vocab.word2id[k], vocab.word2id[c]] = 7
                edge_matrix[vocab.word2id[k], vocab.word2id[c]] = w


        # 缩减知识库的大小
        if KB_percentage > 0:
            logging.info("Keeping {0}% KB concepts...".format(KB_percentage * 100))
            # 随机裁剪 KB_percentage为1的话 edge_matrix不变
            edge_matrix = edge_matrix * (np.random.random((vocab_size, vocab_size)) < KB_percentage).astype(float)

        # 转换为tensor  并to(device)
        edge_matrix = torch.FloatTensor(edge_matrix).to(device)
        relation_matirx = torch.FloatTensor(relation_matirx).to(device)
        # 增加自连接
        edge_matrix[torch.arange(vocab_size), torch.arange(vocab_size)] = 1
        relation_matirx[torch.arange(vocab_size), torch.arange(vocab_size)] = 8#自连接作为第8种关系

    output_size = len(vocab.emotion2id)  # 输出层大小 分类类别数
    max_conversation_length_train = len(train[0])  # 训练集 对话最大长度
    max_conversation_length_val = len(val[0])  # 验证集 对话最大长度

    print("*" * 80)
    logging.info("Number of training utterances: {0}".format(vocab.num_utterances))
    logging.info(
        "Average number of training utterances per conversation: {0}".format(vocab.num_utterances / len(train)))
    logging.info("Max conversation length in training set: {0}".format(max_conversation_length_train))
    logging.info("Max conversation length in validation set: {0}".format(max_conversation_length_val))
    logging.info("Emotion to ids: {0}".format(vocab.emotion2id))
    logging.info("Emotion distribution: {0}".format(vocab.emotion_freq_dist))
    print("*" * 80)

    # 把文本转换为id
    train = convert_examples_to_ids(train, vocab)  # 训练集
    val = convert_examples_to_ids(val, vocab)  #

    # 构建上下文窗口 (一个对话可能对应多个数据)
    train = flatten_examples_classification(train, vocab, k=context_length)
    val = flatten_examples_classification(val, vocab, k=context_length)
    logging.info("Batch size: {0}".format(batch_size))

    # 构建模型 encoder-decoder
    logging.info("Building model...")
    # 模型超参数
    model_kwargs = {
        "src_vocab": vocab_size,
        "N": n_layers,
        "d_model": d_model,
        "d_ff": d_ff,
        "h": h,
        "output_size": output_size,
        "dropout": dropout,
        "KB": bool(KB),
        "context_length": context_length,
    }

    model = make_model(**model_kwargs)

    # 自定义模型参数初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if KB == "conceptnet":
        model.graph_attention.init_params(edge_matrix,relation_matirx)


    logging.info("Initializing pretrained word embeddings into transformer...")
    
    # 使用预训练词向量 初始化词嵌入矩阵
    model.src_embed[0].embedding.weight.data.copy_(torch.from_numpy(pretrained_word_embedding))
    if KB != "":#使用预训练词向量 初始化concept嵌入矩阵
        model.graph_attention.concept_embed.weight.data.copy_(torch.from_numpy(pretrained_word_embedding))
    

    logging.info(model)
    logging.info("Number of model params: {0}".format(count_parameters(model)))

    if args.n_gpu >1:
        model = nn.DataParallel(model)
    model.to(device)

    # weighted crossentropy loss 加权损失
    logging.info("Computing label weights...")
    label_weight = np.array(label_distribution_transformer(val)) / np.array(label_distribution_transformer(train))

    label_weight_binary = np.array([np.sum(label_weight)-label_weight[0],label_weight[0]])

    label_weight = torch.tensor(label_weight / label_weight.sum()).float().to(device) * output_size
    label_weight_binary = torch.tensor(label_weight_binary / label_weight_binary.sum()).float().to(device) * 2


    logging.info("Label weight: {0}".format(label_weight))
    
    criterion = [nn.CrossEntropyLoss(weight=label_weight, reduction="sum"),nn.CrossEntropyLoss(weight=label_weight_binary, reduction="sum"),nn.MSELoss()]
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=1, min_lr=5e-6, verbose=True)

    # training
    train_epoch_losses = []
    val_epoch_losses = []
    logging.info("Start training...")
    best_score = -1
    for epoch in range(1, epochs + 1):
        train_batches = create_batches_classification(train, batch_size, vocab, train=True)
        val_batches = create_batches_classification(val, batch_size, vocab, train=False)

        train_epoch_loss = []  # 存储一个epoch中 各个batch的平均损失
        val_epoch_loss = []
        model.train()

        # model.generator = model.module.generator if args.n_gpu>1 else model.generator
        loss_compute = SimpleLossCompute(model.module if args.n_gpu>1 else model, criterion, dataset, vocab.emotion2id,opt=optimizer,
                                         test=test_mode,lambda1=lambda1,lambda2=lambda2)

        for batch in train_batches:
            batch.to(device)
            out = model.forward(batch.src,
                                batch.src_mask)  # transformer模型输出 (batch_size,d_model)

            loss = loss_compute(out, batch.y, batch.ntokens)  # 和标签计算损失
            train_epoch_loss.append((loss / batch.ntokens).item())

        logging.info("-" * 80)
        logging.info("Epoch {0}/{1}".format(epoch, epochs))
        logging.info("Training loss: {0:.4f}".format(np.mean(train_epoch_loss)))

        train_epoch_losses.append(np.mean(train_epoch_loss))  # 存储每个epoch的损失
        score = loss_compute.score()  # 对一个epoch计算相关指标
        loss_compute.clear()  # 清空

        # 一个epoch结束后 做一次验证
        # get src_attn
        src_attns = []
        model.eval()

        loss_compute = SimpleLossCompute(model.module if args.n_gpu>1 else model, criterion, dataset, vocab.emotion2id,opt=None,
                                         test=test_mode,lambda1=lambda1,lambda2=lambda2)
        with torch.no_grad():
            for batch in val_batches:
                batch.to(device)
                out = model.forward(batch.src,batch.src_mask)
                # get src attn
                loss = loss_compute(out, batch.y, batch.ntokens)
                val_epoch_loss.append((loss / batch.ntokens).item())
            logging.info("Validation loss: {0:.4f}".format(np.mean(val_epoch_loss)))
            val_epoch_losses.append(np.mean(val_epoch_loss))

        # get validation metrics
        score = loss_compute.score()
        if score >best_score:
            best_score = score
        if args.decay==True: #学习率衰减
            scheduler.step(score)
        loss_compute.clear()


    print("final score:")
    print(best_score)

