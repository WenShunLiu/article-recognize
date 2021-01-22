import torch
import torch.nn as nn
import jieba as jb
from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Iterator
from torchtext.data import Example
from torchtext.data import BucketIterator

from util import load_data, datasplit
from net import Net


name_zh = {'LX': '鲁迅', 'MY':'莫言' , 'QZS':'钱钟书' ,'WXB':'王小波' ,'ZAL':'张爱玲'} 
path = "dataset"

# 读取数据，是由tuple组成的列表形式
mydata = load_data(path)

# 定义Field
TEXT  = Field(sequential=True, tokenize=lambda x: jb.lcut(x), lower=True, use_vocab=True)
LABEL = Field(sequential=False, use_vocab=False)
FIELDS = [('text', TEXT), ('category', LABEL)]

# 使用Example构建Dataset
examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), mydata))
dataset = Dataset(examples, fields=FIELDS)

# 构建中文词汇表
TEXT.build_vocab(dataset)

# 生成可迭代的mini-batch
dit = BucketIterator(
    dataset, # 数据集
    batch_size=8,
    #device=-1, # 如果使用gpu，此处将-1更换为GPU的编号
    sort_key=lambda x: len(x.text), 
    sort_within_batch=False,
    repeat=False
)

# 词汇总数
wordCounts = len(TEXT.vocab.stoi)

# 初始化网络
net = Net(wordCounts)

net.load_state_dict(torch.load('results/modelV5.pkl'))

with torch.no_grad():
    data = list(dit)
    total = 0
    correct = 0
    for i in range(len(data)):
        x = data[i].text
        y = data[i].category
        total += y.size(0)
        output = net(x)
        _, predicted = torch.max(output.data, 1)
        for item in range(output.size(0)):
            if(predicted[item] == y[item]):
                correct += 1
    print("正确数：%d\n总数：%d\n准确率：%.4f%%" % (correct, total, 100*correct/total))
    