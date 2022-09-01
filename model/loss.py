import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix, mean_absolute_error, classification_report

from .constants import print_dims

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    def step(self):
        "Update parameter and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * 
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, 
                  torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
            print("{0}: target: type: {1}, shape: {2}".format(self.__class__.__name__, target.type(), target.shape))
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        if print_dims:
            print("{0}: true_dist: type: {1}, shape: {2}".format(self.__class__.__name__, true_dist.type(), true_dist.shape))
        return self.criterion(x, true_dist)


class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, model, criterion, dataset, emotion2id,opt=None, test=False,lambda1=1.0,lambda2=1.0):
        self.model = model #输出层
        self.criterion = criterion #交叉熵损失函数
        self.opt = opt #Adam
        self.dataset = dataset #当前数据集
        self.emotion2id = emotion2id #标签到索引的映射字典   索引0对应出现频率最高的标签 中立标签
        self.test = test #模式
        self.outputs = []
        self.tgts = []
        self.lambda1=lambda1
        self.lambda2=lambda2
    @staticmethod
    def compute_score(predictions, ground, dataset, emotion2id, test=False):
        pred_y = predictions.astype(int) #模型的预测标签 （一个epoch）
        val_y = ground.astype(int)   #真实标签 （一个epoch）
        
        if dataset in ["EC"]: #EC数据集
            labels = [emotion2id["happy"], emotion2id["angry"], emotion2id["sad"]]#提取情感标签 不包括中立或others
            score = f1_score(val_y, pred_y, average='micro', labels=labels) #只对情感标签 计算micro f1-score
            print("Micro-F1 (exclude neutral): {0}".format(score))
            if test: #如果是测试模式 打印分类报告（只包括情感标签）
                print(classification_report(val_y, pred_y, labels=labels, digits=4))
            cm = confusion_matrix(val_y,pred_y)
            print(cm)

        elif dataset in ["DD"]:
            labels = [emotion2id[str(i)] for i in range(1, 7)]#提取情感标签 不包括中立或others="0"
            score = f1_score(val_y, pred_y, average='micro', labels=labels)#只对情感标签 计算micro f1-score
            print("Micro-F1 (exclude neutral): {0}".format(score))
            if test:#如果是测试模式 打印分类报告（只包括情感标签）
                print(classification_report(val_y, pred_y, labels=labels, digits=4))
            
            cm = confusion_matrix(val_y,pred_y)
            print(cm)
        elif dataset in ["MELD", "IEMOCAP", "EmoryNLP"]: #剩余的这三个数据集 各个类别比较均衡 对所有的类别，包括中立类别，计算weighted f1-score
            score = f1_score(val_y, pred_y, average='weighted')
            print("Weighted Macro-F1: {0}".format(score))
            if test:#如果是测试模式 打印分类报告
                print(classification_report(val_y, pred_y, digits=4))
            
            cm = confusion_matrix(val_y,pred_y)
            print(cm)
        else: #对应一些标签不是离散值的数据集 如AVEC数据集
            score = mean_absolute_error(val_y, pred_y)
            print("MAE: {0}".format(score))
            if test:
                print(mean_absolute_error(val_y, pred_y))
        return score


    def score(self):
        score = self.compute_score(np.array(self.outputs), np.array(self.tgts), self.dataset, self.emotion2id, test=self.test)
        return score


    def clear(self):
        self.outputs = []
        self.tgts = []
        
    def __call__(self, x, y, norm):
        """
        x: (batch_size, tgt_seq_len, d_model)
        y: (batch_size, tgt_seq_len)
        norm: ()
        """
        if print_dims:
            print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
            print("{0}: y: type: {1}, shape: {2}".format(self.__class__.__name__, y.type(), y.shape))
            print("{0}: norm: type: {1}, shape: {2}".format(self.__class__.__name__, norm.type(), norm.shape))
        x1,x2,x3 = self.model.generator(x) #把transformer的输出 通过输出层  得到最后的输出 (batch_size,C)

        label1 = y
	#基于原始情感分类任务的label 构建两个辅助任务的label
        label3 = torch.zeros(x.size(0), x3.size(-1))
        label2 = [int(i == 0) for i in y]

        for i in range(len(y)):
            if y[i] > 0:
                label3[i][y[i] - 1] = 1
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        label2 = torch.LongTensor(label2).to(device)
        label3 = torch.Tensor(label3).to(device)



        loss1 = self.criterion[0](x1.contiguous().view(-1, x1.size(-1)), label1.contiguous().view(-1)) / norm #一个batch的平均（加权）损失
        loss2 = self.criterion[1](x2.contiguous().view(-1,x2.size(-1)),label2.contiguous().view(-1)) /norm
        loss3 = self.criterion[2](x3.contiguous().view(-1,x3.size(-1)),label3.contiguous().view(-1, x3.size(-1)))

        self.outputs += x1.contiguous().view(-1, x1.size(-1)).argmax(dim=-1).tolist() #模型的预测标签 (batch_size,)
        self.tgts += label1.contiguous().view(-1).tolist() #真实标签 (batch_size,)

        loss = (loss1 +  self.lambda1* loss2 +self.lambda2* loss3) / float(1 + self.lambda1+self.lambda2) 
        #loss = (loss1 +  self.lambda1* loss2 +self.lambda2* loss3) 
        if print_dims:
            print("{0}: loss: type: {1}, shape: {2}".format(self.__class__.__name__, loss.type(), loss.shape))
        
        if self.opt is not None:
            loss.backward()     #计算梯度
            self.opt.step()  #反向传播
            if isinstance(self.opt, NoamOpt): #清空梯度
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.item() * norm  #一个batch的总损失
