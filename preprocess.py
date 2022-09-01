import spacy
import argparse
from tqdm import tqdm
import pandas as pd
import pickle
import random
import json
from functools import partial
from collections import Counter

from utils.tools import counter_to_distribution

spacy_en = spacy.load("en")


def create_examples_EC(split):
    """
        split: train, val or test
        return: a list of examples, each example is a list of utterances and an emotion label for the last utterance
    """
    #EC数据集预处理  分别创建训练集、验证集、测试集
    with open("./data/EC/{0}.txt".format(split), "r") as f:
        f.readline()    #跳过表头 id turn1 turn2 turn3 label
        conversations = f.readlines() #读取剩余所有行/对话（3轮）
    #每个数据集 拥有的对话数
    print("{0} split has {1} conversations".format(split, len(conversations)))
    #对话长度3   3轮
    print("max_conv_length: ", 3)
    
    examples = []
    #第一个对话 对应的标签
    dummy_emotion = conversations[0].strip().split("\t")[-1]
    for conv in conversations: #遍历所有对话（3轮） 一个对话对应一个样本/数据
        #每个对话的内容（3个子句）和标签
        utterances_emotion = [e.strip() for e in conv.split("\t")][1:]
        ex = []
        #遍历每个子句 两人对话 交替进行 区分说话人
        for idx, utter in enumerate(utterances_emotion[:-1]):
            if idx%2 == 0:
                speaker = "Speaker A"
            else:
                speaker = "Speaker B"
            if idx <= 1: #上下文 统一标签为dummy_emotion
                ex.append((utter, speaker, dummy_emotion, 0))
            elif idx == 2: #最后一个子句
                ex.append((utter, speaker, utterances_emotion[-1], 1))
        examples.append(ex) #[[(utter1,speakerA,dummy_emotion,0),(utter2,speakerB,dummy_emotion,0),(utter3,speakerA,label,1)],[(),(),()]...]
    return examples


def create_examples_DD(split):
    """
        split: train, val or test
        return: a list of examples, each example is a list of utterances and an emotion label for the last utterance
    """
    # DailyDialogue数据集预处理  分别创建训练集、验证集、测试集

    #对话文件
    with open("./data/DD/dialogues_{0}.txt".format(split), "r") as f:
        conversations = f.readlines() #每一行代表一个对话 包含多个子句，每个对话包含的子句数量不一定
    #对话标签文件
    with open("./data/DD/dialogues_emotion_{0}.txt".format(split), "r") as f:
        emotions = f.readlines() #每一行表示 一个对话中各个子句 对应的标签（每个子句都有）
    print("{0} split has {1} conversations".format(split, len(conversations)))
    
    examples = []
    #最长对话的长度 包含的子句数. 对话中每个子句之间用__eou__分隔 [:-1]不包括最后一个，因为对话末尾有一个__eou__，因此最后一个为空白符
    max_conv_length = max([len(conv.split("__eou__")[:-1]) for conv in conversations])
    print("max_conv_length: ", max_conv_length)
    
    #做填充用
    dummy_utterance = "this is a dummy sentence"
    dummy_emotion = emotions[0].strip().split(" ")[0] #第一个对话的第一个子句对应的标签
    dummy_speaker = "Speaker A"

    for conv, emo in zip(conversations, emotions): #成对遍历 对话和标签
        utterances = [utter.strip() for utter in conv.split("__eou__")][:-1]  #把对话切分为子句列表
        
        # 添加说话人信息  二人对话，交替进行
        speakers = []
        for idx, utter in enumerate(utterances): #遍历子句
            if idx%2 == 0:
                speaker = "Speaker A"
            else:
                speaker = "Speaker B"
            speakers.append(speaker)
        
        # 添加每个子句对应的标签
        utter_emotions = [emotion.strip() for emotion in emo.split(" ")[:-1]]
        assert len(utterances) == len(speakers) == len(utter_emotions)
                
        # 用dummy进行填充至最长的对话长度
        conv_length = len(utterances) #对话长度 包含的子句数
        masks = conv_length*[1] + (max_conv_length-conv_length)*[0] #填充部分的mask对应0
        utterances += (max_conv_length-conv_length)*[dummy_utterance] #填充部分的子句 对应dummy_utterance
        utter_emotions += (max_conv_length-conv_length)*[dummy_emotion] #填充部分子句对应的标签为dummy_emotion
        speakers += (max_conv_length-conv_length)*[dummy_speaker] #填充部分子句对应的说话人为dummy_speaker
        
        #创建样本 一个完整的对话对应一个样本
        examples.append(list(zip(utterances, speakers, utter_emotions, masks))) #[[(),(),...],...]
    return examples


def create_examples_MELD(split):
    """
        split: train, val or test
        return: a list of examples, each example is a list of utterances and an emotion label for the last utterance
    """
    data = pd.read_csv("./data/MELD/{0}.csv".format(split))
    print("{0} split has {1} conversations".format(split, data["Dialogue_ID"].unique().shape[0]))
    
    examples = []
    conv_lengths = []
    for idx, conv in data.groupby("Dialogue_ID"):
        conv_lengths.append(len(conv))
    max_conv_length = max(conv_lengths)
    print("max_conv_length: ", max_conv_length)
    
    dummy_utterance = "this is a dummy sentence"
    dummy_emotion = data["Emotion"][0]
    dummy_speaker = data["Speaker"][0]
    for idx, conv in data.groupby("Dialogue_ID"):
        utterances = conv["Utterance"].tolist()
        speakers = conv["Speaker"].tolist()
        emotions = conv["Emotion"].tolist()
        
        assert len(utterances) == len(speakers) == len(emotions)
                
        # pad dummy utterances
        conv_length = len(utterances)
        masks = conv_length*[1] + (max_conv_length-conv_length)*[0]
        utterances += (max_conv_length-conv_length)*[dummy_utterance]
        emotions += (max_conv_length-conv_length)*[dummy_emotion]
        speakers += (max_conv_length-conv_length)*[dummy_speaker]
        
        # associate utterances with speakers
        utterances = list(zip(utterances, speakers, emotions, masks))

        examples.append(utterances)
    return examples


def create_examples_EmoryNLP(split):
    """
        split: train, val or test
        return: a list of examples, each example is a list of utterances and an emotion label for the last utterance
    """
    data = pd.read_csv("./data/EmoryNLP/{0}.csv".format(split))
    print("{0} split has {1} conversations".format(split, data[["Season", "Episode", "Scene_ID"]].drop_duplicates().shape[0]))
    
    examples = []
    conv_lengths = []
    for idx, conv in data.groupby(["Season", "Episode", "Scene_ID"]):
        conv_lengths.append(len(conv))
    max_conv_length = max(conv_lengths)
    print("max_conv_length: ", max_conv_length)
    
    dummy_utterance = "this is a dummy sentence"
    dummy_emotion = data["Emotion"][0]
    dummy_speaker = data["Speaker"][0][2:-2]
    for idx, conv in data.groupby(["Season", "Episode", "Scene_ID"]):
        utterances = conv["Utterance"].tolist()
        speakers = [speaker[2:-2] for speaker in conv["Speaker"].tolist()]
        emotions = conv["Emotion"].tolist()
        
        assert len(utterances) == len(speakers) == len(emotions)
                
        # pad dummy utterances
        conv_length = len(utterances)
        masks = conv_length*[1] + (max_conv_length-conv_length)*[0]
        utterances += (max_conv_length-conv_length)*[dummy_utterance]
        emotions += (max_conv_length-conv_length)*[dummy_emotion]
        speakers += (max_conv_length-conv_length)*[dummy_speaker]
        
        # associate utterances with speakers
        utterances = list(zip(utterances, speakers, emotions, masks))

        examples.append(utterances)
    return examples
    


'''
# Regarding producing data.pkl for IEMOCAP:
# The exact code for preprocessing IEMOCAP raw to data.pkl is left somewhere in my older server
# The following code is not gurrenteed to work and just intended to give some clues.

dataset = pickle.load(open("./data/IEMOCAP/IEMOCAP_features_raw.pkl", 'rb'), encoding='latin1')
# randomly select validation sessions
train_ids = list(dataset[7])
val_ids = []
num_vals = 20

while len(set(val_ids)) < num_vals:
    random_id = random.choice(train_ids)
    val_ids.append(random_id)
val_ids = list(set(val_ids))
train_ids = list(set(train_ids) - set(val_ids))
test_ids = list(dataset[8])

new_dataset = []
new_dataset.append(dataset[1]) # speaker
new_dataset.append(dataset[6]) # utterances
new_dataset.append(dataset[2]) # label
new_dataset.append(train_ids) 
new_dataset.append(val_ids)
new_dataset.append(test_ids)
    
# save new dataset
with open("./data/IEMOCAP/data.pkl", "wb") as f:
    pickle.dump(new_dataset, f)
'''



def create_examples_IEMOCAP(split):
    dataset = pickle.load(open("./data/IEMOCAP/data.pkl", 'rb'))
    if split == "train":
        session_ids = dataset[3]
    elif split == "val":
        session_ids = dataset[4]
    else:
        session_ids = dataset[5]
    print("{0} split has {1} conversations".format(split, len(session_ids)))
        
    examples = []
    conv_lengths = []
    for i in session_ids:
        conv_lengths.append(len(dataset[1][i]))
        dummy_emotion = dataset[2][i][0]
        dummy_speaker = dataset[0][i][0]
    max_conv_length = max(conv_lengths)
    print("max_conv_length: ", max_conv_length)
    
    dummy_utterance = "this is a dummy sentence"
    for i in session_ids:
        speakers = dataset[0][i]
        utterances = dataset[1][i]
        emotions = dataset[2][i]
        assert len(speakers) == len(utterances) == len(emotions)
        
        # pad dummy utterances
        conv_length = len(utterances)
        masks = conv_length*[1] + (max_conv_length-conv_length)*[0]
        utterances += (max_conv_length-conv_length)*[dummy_utterance]
        emotions += (max_conv_length-conv_length)*[dummy_emotion]
        speakers += (max_conv_length-conv_length)*[dummy_speaker]
        
        # associate utterances with speakers
        utterances = list(zip(utterances, speakers, emotions, masks))

        examples.append(utterances)
            
    return examples


def clip_conversation_length(examples, max_conversation_length):
    """
        examples: a list of examples
        max_conversation_length: the max number of utterances in one example
        return: a list of clipped examples where each example is limited to the most recent k utterances
    """
    clipped_examples = []
    num_clips = 0
    for ex in examples:
        if len(ex) > max_conversation_length+1:
            num_clips += 1
            ex = ex[-(max_conversation_length+1):]
        clipped_examples.append(ex)
    print("Number of clipped examples: {0}".format(num_clips))
    return clipped_examples


def clean(text, max_sequence_length):
    """
        text: a piece of text in str
        max_sequence_length: the max sequence length for each utterance
        return: a list tokenized cleaned words
    """
    #文本小写化
    text = text.lower()
    
    #可以添加其他预处理过程
    
    #分词 长截短填
    return [token.text for token in spacy_en.tokenizer(text)][:max_sequence_length]


def clean_examples(examples, max_sequence_length):
    """
        examples: a list of examples, each example is a list of (utterance, speaker, emotion, mask)
        max_sequence_length: the max sequence length for each utterance
        return: a list tokenized cleaned examples
    """
    cleaned_examples = []
    for ex in tqdm(examples):
        cleaned_examples.append([(clean(utterance, max_sequence_length), speaker, emotion, mask) 
                                 for utterance, speaker, emotion, mask in ex])
    return cleaned_examples


#预处理字典 键为数据集名称，值对应数据集的预处理函数
create_examples_dict = {
    "EC": create_examples_EC,
    "DD": create_examples_DD,
    "MELD": create_examples_MELD,
    "EmoryNLP": create_examples_EmoryNLP,
    "IEMOCAP": create_examples_IEMOCAP
}


if __name__ == "__main__":

    #数据预处理
    #声明 argparse 对象
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    #添加命令行参数
    #必选参数  待处理数据集
    parser.add_argument('--dataset', help='Dataset to preprocess', choices=['EC','DD','MELD', "EmoryNLP", "IEMOCAP"], required=True)
    #对话最大长度  包含的子句数
    parser.add_argument('--max_conversation_length', type=int, default=10)
    #每个子句的最大长度 长截短填
    parser.add_argument('--max_sequence_length', type=int, default=30)
    
    #解析参数
    args = parser.parse_args()

    #把命令行参数 赋给全局变量
    dataset = args.dataset
    max_conversation_length = args.max_conversation_length
    max_sequence_length = args.max_sequence_length
    
    # 创建样本
    print("Preprocessing {0}...".format(dataset))

    #获取所选数据集对应的预处理函数
    create_examples = create_examples_dict[dataset]

    for split in ["train", "val", "test"]: #分别创建训练、验证、测试集
        examples = create_examples(split)
        # examples = clip_conversation_length(examples, max_conversation_length)
        #清洗 填充
        examples = clean_examples(examples, max_sequence_length)
        
        #把预处理好的数据 存储为.pkl文件
        path_to_save = "./data/{0}/{1}.pkl".format(dataset, split)
        print("Saving data to {0}".format(path_to_save))
        with open(path_to_save, "wb") as f:
            pickle.dump(examples, f)
