import csv
import argparse
from ast import literal_eval
from collections import defaultdict
from utils.io import load_pickle, to_pickle
import os


def get_ngrams(utter, n):
    # utter: 子句 已分好词
    # n: up to n-grams
    total = []
    for i in range(len(utter)):
        for j in range(i, max(i - n, -1), -1):
            total.append("_".join(utter[j:i + 1]))
    return total


# get all ngrams for a dataset
def get_all_ngrams(examples, n):
    all_ngrams = []
    for ex in examples:  # 遍历每个对话
        for utter, _, _, _ in ex:  # 遍历每个子句
            all_ngrams.extend(get_ngrams(utter, n))
    return set(all_ngrams)  # 去重


if __name__ == "__main__":

    # 声明argparse对象
    parser = argparse.ArgumentParser()
    # 添加参数
    # 必选参数 数据集
    parser.add_argument('--dataset', required=True)
    # n-gram  默认为1 只考虑一个词 不考虑组合
    parser.add_argument('--n', default=1)
    # 解析参数
    args = parser.parse_args()

    # 把命令行参数 赋给全局变量
    dataset = args.dataset
    n = args.n

    print("Loading dataset...")
    # 加载之前处理好的 相应数据集的.pkl文件
    train = load_pickle("./data/{0}/{1}.pkl".format(dataset, "train"))
    val = load_pickle("./data/{0}/{1}.pkl".format(dataset, "val"))
    test = load_pickle("./data/{0}/{1}.pkl".format(dataset, "test"))

    ngrams = get_all_ngrams(train + val + test, n)  # 合并训练、验证、测试集  得到所有子句的n-gram（默认为1-gram，单个词）并去重

    # 对于每个n-gram/词 获取他在concepnet中的直接邻居
    print("Loading conceptnet...")  # 加载conceptnet
    csv_reader = csv.reader(open("./data/KB/5.6.1/conceptnet-assertions-5.6.0.csv", "r"), delimiter="\t")
    concept_dict = defaultdict(set)

    for i, row in enumerate(csv_reader):  # 遍历csv文件每一行
        if i % 1000000 == 0:
            print("Processed {0} rows".format(i))

        lang = row[2].split("/")[2]  # 获取语言
        if lang == 'en':  # 如果是英文
            # 三元组
            relation = row[1].split("/")[2] #关系
            if relation == 'dbpedia':
                continue
            c1 = row[2].split("/")[3]  # concept1
            c2 = row[3].split("/")[3]  # concept2
            weight = literal_eval(row[-1])["weight"]  # 置信度
            if c1 in ngrams:
                concept_dict[c1].add((c2, weight,relation))
            if c2 in ngrams:
                concept_dict[c2].add((c1, weight,relation))

    print("Saving concepts...")  # 把每个数据集对应的预处理好的conceptnet 保存为.pkl文件
    to_pickle(concept_dict, "./data/KB/5.6.1/{0}.pkl".format(dataset))


