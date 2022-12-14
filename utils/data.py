import re
import csv
import random
from collections import Counter, defaultdict

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords


DD_label_to_emotions = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"}
IEMOCAP_emotions_to_label = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
IEMOCAP_label_to_emotions = {0: 'hap', 1: 'sad', 2: 'neu', 3: 'ang', 4: 'exc', 5: 'fru'}
nltk_stopwords = stopwords.words('english')
# spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS # older version of spacy
stopwords = set(nltk_stopwords).union(spacy_stopwords)
porter = PorterStemmer()

class Vocab(object):
    def __init__(self, examples, min_freq, max_vocab_size):
        """
            examples: a list of examples, each example is a list of (utter, speaker, emotion, mask), followed by an emotion label
            min_freq: the min frequency of word in the vocabulary
            max_vocab_size: max vocabulary size

            return: a Vocab object
        """
        self.min_freq = min_freq #最小词频
        self.max_vocab_size = max_vocab_size #最大词典大小

        # 添加填充符pad 和 未知符 unk
        self.word2id = {"<pad>": 0, "<unk>": 1}
        self.id2word = {0: "<pad>", 1: "<unk>"}

        self.speaker2id = {}
        self.emotion2id = {}

        self.word_freq_dist = Counter()
        self.speaker_freq_dist = Counter()
        self.emotion_freq_dist = Counter()
        
        utterance_lengths = []
        conversation_lengths = []

        for ex in examples: #遍历每个对话
            conv_length = sum([utter[-1] for utter in ex]) #对话长度 非填充的有真实标签的子句数量（EC为1，EC数据集不需要填充，每个对话固定有3个子句，且只有最后一个子句有真实标签）
            conversation_lengths.append(conv_length) #统计对话的长度
            for utter, speaker, emotion, mask in ex: #遍历每个子句
                if mask == 1: #如果子句不是填充的且有真实标签 （EC数据集的话是对后一个子句）
                    self.word_freq_dist.update(utter) #统计词频
                    self.speaker_freq_dist.update([speaker])#统计说话者频率
                    self.emotion_freq_dist.update([emotion]) #统计标签频率
                    utterance_lengths.append(len(utter)) #统计子句的长度

        # 过滤低频词
        words = [(w,cnt) for w, cnt in self.word_freq_dist.items() if cnt >= min_freq]
        words = sorted(words, key=lambda x: x[1], reverse=True) #按词频从大到小排序 [(word,freq),...]

        #截断为设定的最大词典大小
        words = words[:max_vocab_size]

        # 构建词到索引 以及 索引到词的映射
        for idx, (w, cnt) in enumerate(words):
            self.word2id[w] = idx+2
            self.id2word[idx+2] = w

        # 按说话者出现的频率 从大到小排序
        speakers = sorted(self.speaker_freq_dist.items(), key=lambda x: x[1], reverse=True)
        #构建说话者到id的映射
        for idx, (speaker, cnt) in enumerate(speakers):
            self.speaker2id[speaker] = idx

        #按情感出现的频率 从大到小排序
        emotions = sorted(self.emotion_freq_dist.items(), key=lambda x: x[1], reverse=True)
        # 构建情感到id的映射
        for idx, (emotion, cnt) in enumerate(emotions):
            self.emotion2id[emotion] = idx


        
        self.max_conversation_length = max(conversation_lengths) #最大的对话长度（非填充且有真实标签的子句数量，EC为1）
        self.max_sequence_length = max(utterance_lengths) #最大的子句长度（非填充且有真实标签）
        self.num_utterances = len(utterance_lengths) #子句数量 （非填充且有真实标签，EC中等于对话数量）
        

def convert_examples_to_ids(examples, vocab):
    """
        examples: a list of examples, each example is a list of (utter, speaker, emotion, mask), followed by an emotion label
        vocab: a Vocab object
        max_sequence_length: max sequence length
        
        return: examples containing word ids
    """
    #把单词 转换为 id
    examples_ids = []
    for ex in examples: #遍历每个对话
        ex_ids = []
        for utter, speaker, emotion, mask in ex: #遍历每个子句
            #把子句中的单词 转换为词典中的索引 不在词典中的转换为unk对应的索引
            utter_ids = [vocab.word2id[w] if w in vocab.word2id else vocab.word2id["<unk>"] for w in utter]
            #填充部分转换为pad对应的索引
            utter_ids = utter_ids + (vocab.max_sequence_length - len(utter_ids)) * [vocab.word2id["<pad>"]]
            #把说话人 也转换为对应的id
            if speaker in vocab.speaker2id:
                speaker_id = vocab.speaker2id[speaker]
            else:
                speaker_id = len(vocab.speaker2id) # val or test set, use a new speaker id
            emotion_id = vocab.emotion2id[emotion] #把标签转换为id
            ex_ids.append((utter_ids, speaker_id, emotion_id, mask))
        examples_ids.append(ex_ids)
    return examples_ids

def create_one_batch(examples):
    """
        examples: a batch of examples having the same number of turns and seq_len, each example is a list of (utter, speaker, label, mask)
        
        return: batch_x, batch_y, where batch_x is a tuple of token ids and speaker ids
    """
    tokens = []
    speakers = []
    labels = []
    masks = []
    for ex in examples:
        ex_tokens, ex_speakers, ex_labels, ex_masks = list(zip(*ex))
        tokens.append(ex_tokens)
        speakers.append(ex_speakers)
        labels.append(ex_labels)
        masks.append(ex_masks)
    
    return (tokens, speakers), labels, masks


def create_batches(examples, batch_size, train=True):
    batch_data = []
    if train == True:
        random.shuffle(examples)
    
    batch_ids = list(range(0, len(examples), batch_size)) + [len(examples)]
    for s, e in zip(batch_ids[:-1], batch_ids[1:]):
        batch_examples = examples[s:e]
        batch_x, batch_y, batch_mask = create_one_batch(batch_examples)
        batch_data.append((batch_x, batch_y, batch_mask))
    return batch_data


def create_balanced_batches(examples, batch_size, train=True):
    batch_data = []
    if train == True:
        random.shuffle(examples)
    
    num_valid_utterances = []
    for ex in examples:
        num_valid_utterances.append(sum([mask for utter, speaker, label, mask in ex]))
    
    batch_ids = [0]
    for i in range(len(examples)):
        if len(num_valid_utterances[batch_ids[-1]:i]) >= batch_size:
            batch_ids.append(i)
    
    for s, e in zip(batch_ids[:-1], batch_ids[1:]):
        batch_examples = examples[s:e]
        batch_x, batch_y, batch_mask = create_one_batch(batch_examples)
        batch_data.append((batch_x, batch_y, batch_mask))
    return batch_data


def merge_splits(train, val):
    if len(train[0]) > len(val[0]):
        num_additional_utterances = len(train[0]) - len(val[0])
        for ex in val:
            if ex[-1][0] == ['this', 'is', 'a', 'dummy', 'sentence']:
                dummy_ex = ex[-1]
                break
        new_val = []
        for ex in val:
            new_val.append(ex + num_additional_utterances*[dummy_ex])
        return train + new_val
    elif len(train[0]) < len(val[0]):
        num_additional_utterances = len(val[0]) - len(train[0])
        for ex in train:
            if ex[-1][0] == ['this', 'is', 'a', 'dummy', 'sentence']:
                dummy_ex = ex[-1]
                break
        new_train = []
        for ex in train:
            new_train.append(ex + num_additional_utterances*[dummy_ex])
        return new_train + val
    else:
        return train + val


def get_vocab_embedding(vocab, vectors, embedding_size):
    pretrained_word_embedding = np.zeros((len(vocab.word2id), embedding_size)) #初始化词嵌入矩阵为0
    for w, i in vocab.word2id.items(): #对存在预训练词向量的词 用预训练词向量 对词嵌入矩阵该词对应的未知进行赋值
        pretrained_word_embedding[i] = vectors.query(w)
    return pretrained_word_embedding #返回词嵌入矩阵


# 构建边矩阵
def filter_conceptnet(conceptnet, vocab):
    filtered_conceptnet = {}
    for k in conceptnet: #遍历conceptnet中的每个concept
        if k in vocab.word2id and k not in stopwords: #concept要在相应数据集的词典中，且不是停止词
            filtered_conceptnet[k] = set()
            for c,w,r in conceptnet[k]: #某个concept在conceptnet中对应的另一个concept 以及他们之间的置信度[0,10]和关系
                if c in vocab.word2id and c not in stopwords and w>=1: #另一个concept要在词典中，且置信度要>=1
                    filtered_conceptnet[k].add((c,w,r))

    return filtered_conceptnet #返回过滤后的conceptnet


# 去除相同的concept有多个不同的置信度的情况，保留一个最大的置信度。 （因为我们忽略了conceptnet中边上的关系，简单把其看作一个同构图，所以会出现一对相同的concept之间有多个不同的置信度的情况，实际上他们之间的关系不同）
def remove_KB_duplicates(conceptnet):
    filtered_conceptnet = {}
    for k in conceptnet:
        filtered_conceptnet[k] = set()
        concepts = set()
        filtered_concepts = sorted(conceptnet[k], key=lambda x: x[1], reverse=True)
        for c,w,r in filtered_concepts:
            if c not in concepts:
                filtered_conceptnet[k].add((c, w,r))
                concepts.add(c)
    return filtered_conceptnet
