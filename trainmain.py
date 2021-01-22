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
import matplotlib.pyplot as plt

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

# 词汇总数
wordCounts = len(TEXT.vocab.stoi)

# 初始化网络
net = Net(wordCounts)

lossf = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.Adam(net.parameters())

losslist = []
for epoch in range(40):
    for idx, batch in enumerate(train_iter):
        optimizer.zero_grad()
        x = batch.text
        y = batch.category
        output = net(x)
        loss = lossf(output, y)
        loss.backward()
        optimizer.step()
        if(idx % 10 == 0):
            losslist.append(loss.item())
            print("[%d, %d] loss: %.10f"%(epoch+1, idx, loss.item()))

torch.save(net.state_dict(), 'results/modelV5.pkl')
plt.plot(losslist, label = "loss")
plt.legend()
plt.show()