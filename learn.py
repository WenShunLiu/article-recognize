# 导入相关包
import copy
import os
import random
import numpy as np
import jieba as jb
import jieba.analyse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as f
from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Iterator
from torchtext.data import Example
from torchtext.data import BucketIterator

from util import load_data, datasplit



name_zh = {'LX': '鲁迅', 'MY':'莫言' , 'QZS':'钱钟书' ,'WXB':'王小波' ,'ZAL':'张爱玲'} 
path = "dataset"

# 读取数据，是由tuple组成的列表形式
mydata = load_data(path)[:100]

# 定义Field
TEXT  = Field(sequential=True, tokenize=lambda x: jb.lcut(x), lower=True, use_vocab=True)
LABEL = Field(sequential=False, use_vocab=False)
FIELDS = [('text', TEXT), ('category', LABEL)]



# 使用Example构建Dataset
examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), mydata))
dataset = Dataset(examples, fields=FIELDS)

# 构建中文词汇表
TEXT.build_vocab(dataset)

idx2word = [w for w, i in dict(TEXT.vocab.stoi).items()]

td, vd = dataset.split(split_ratio=0.7)

# 生成可迭代的mini-batch
train_iter, val_iter = BucketIterator.splits(
    (td, vd), # 数据集
    batch_sizes=(8, 8),
    #device=-1, # 如果使用gpu，此处将-1更换为GPU的编号
    sort_key=lambda x: len(x.text), 
    sort_within_batch=False,
    repeat=False
)

b = [[idx2word[idx] for idx in v] for v in list(train_iter)[0].text]
print(b)
print(list(train_iter)[0].text)
print(list(train_iter)[0].text.size())


embeds = nn.Embedding(len(idx2word), 5)
wv = embeds(list(train_iter)[0].text)
print(wv.size())
print(wv)

text = [[b[i][v] for i in range(len(b))] for v in range(8)]
print(text)