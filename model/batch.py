import numpy as np
import torch

# no tgt mask for classification
def no_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size) # size: seq_len
    subsequent_mask = np.zeros(attn_shape).astype('uint8')
    # subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # print("subsequent mask shape: ", subsequent_mask.shape) # (1, seq_len-1, seq_len-1)
    return torch.from_numpy(subsequent_mask) == 0


class ClassificationBatch:
    "Object for holding a batch of data with mask during training"
    def __init__(self, src, label, pad=0, concept=None):

        self.src = src #(batch_size,(context_length+1)*seq_len)
        self.src_mask = (src != pad).unsqueeze(-2) #计算注意力时要用到 mask encoder端

        self.concept = concept
        self.concept_mask = None

        if concept is not None:
            self.concept_mask = (concept != pad).unsqueeze(-2)

        self.y = label  #(batch_size)
        # self.tgt_mask = self.make_std_mask(self.tgt, pad)
        self.ntokens = torch.FloatTensor([len(src)])[0] # batch_size
        
    def to(self, device):
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)

        if self.concept is not None:
            self.concept = self.concept.to(device)
            self.concept_mask = self.concept_mask.to(device)
        self.y = self.y.to(device)
        self.ntokens = self.ntokens.to(device)
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words"
        tgt_mask = (tgt != pad).unsqueeze(-2) # (batch_size, 1, seq_len-1)
        # print("tgt_mask shape: ", tgt_mask.shape)
        tgt_mask = tgt_mask & no_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask


def flatten_examples_classification(examples, vocab, k=1):
    # returns a list of ([(utter1, speaker1), (utter2, speaker2), ..., (utterk, speakerk)], label)
    import random
    classfication_examples = []
    all_speakers = list(vocab.speaker2id.values())
    empty_ex = len(examples[0][0][0])*[vocab.word2id["<pad>"]] #空子句
    for ex in examples:  #遍历每个对话
        for i in range(len(ex)): #遍历每个子句
            mask = ex[i][-1]
            if mask == 1:#有真实标签 非填充的子句
                if i < k: #小于上下文窗口大小 则进行填充
                    context = [(empty_ex.copy(), random.choice(all_speakers)) for i in range(k-i)] # k-i
                    context += [(ex[i-j][0].copy(), ex[i-j][1]) for j in range(i, 0, -1)] # (k-i) + (i) = k
                else:
                    context = [(ex[i-j][0].copy(), ex[i-j][1]) for j in range(k, 0, -1)] # k
                if k > 0:
                    new_ex = (context + [(ex[i][0].copy(), ex[i][1])], ex[i][2]) #context 再加上最后一个子句
                else:
                    new_ex = [(empty_ex.copy(), random.choice(all_speakers)), (ex[i][0].copy(), ex[i][1])], ex[i][2] # add one empty context utterance
                # if i==0:
                #     random_speaker = random.choice(all_speakers)
                #     new_ex = [(empty_ex.copy(), random_speaker), (ex[i][0].copy(), ex[i][1])], ex[i][2]
                # else:
                #     new_ex = [(ex[i-1][0].copy(), ex[i-1][1]), (ex[i][0].copy(), ex[i][1])], ex[i][2]
                classfication_examples.append(new_ex)
    return classfication_examples


def create_batches_classification(examples, batch_size, vocab, train=True):
    import random
    
    def create_one_batch(examples, vocab):
        """
            examples: a batch of examples having the same number of turns and seq_len, 
                each example is a list of ([(utter1, speaker1), ..., (utterk, speakerk), (utterA, speakerA)], label)

            return: batch_Q, batch_Q_speakers, batch_A, batch_A_speakers, batch_label
        """
        Qs = [] # context+current utterances
        Q_speakers = [] # context+current speakers

        labels = [] # label of current utterance

        for ex in examples:
            context = []
            context_speakers = []
            for Q, Q_speaker in ex[0]:
                context.extend(Q)
                context_speakers.append(Q_speaker)

            label = ex[1]
            
            Qs.append(context)
            Q_speakers.append(context_speakers)

            labels.append(label)
        
        batch = ClassificationBatch(torch.LongTensor(Qs), torch.LongTensor(labels), vocab.word2id["<pad>"])
        return batch

    batch_data = []

    if train == True:
        random.shuffle(examples) #打乱训练集

    batch_ids = list(range(0, len(examples), batch_size)) + [len(examples)]  #[0,0+batch_size,0+2*batch_size,...,len(examples)]
    for s, e in zip(batch_ids[:-1], batch_ids[1:]): # 遍历每个batch的起点和终点
        batch_examples = examples[s:e] #一个batch的数据
        one_batch = create_one_batch(batch_examples, vocab) #创建单个batch
        batch_data.append(one_batch) #存储每个batch到一个列表中
    return batch_data
