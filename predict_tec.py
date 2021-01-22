import torch
import torch.nn as nn
import jieba as jb
from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Example

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

# ----------------------------- 请加载您最满意的模型 -------------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 temp.pth 模型，则 model_path = 'results/temp.pth'

# 创建模型实例
wordCounts = len(TEXT.vocab.stoi)
model = Net(wordCounts)
model_path = "results/modelV2.pkl"
model.load_state_dict(torch.load(model_path))

# -------------------------请勿修改 predict 函数的输入和输出-------------------------
def predict(text):
    """
    :param text: 中文字符串
    :return: 字符串格式的作者名缩写
    """
     # ----------- 实现预测部分的代码，以下样例可代码自行删除，实现自己的处理方式 -----------
    labels = {0: 'LX', 1: 'MY', 2: 'QZS', 3: 'WXB', 4: 'ZAL'}
    # 自行实现构建词汇表、词向量等操作
    # 将句子做分词，然后使用词典将词语映射到他的编号
    text2idx = [TEXT.vocab.stoi[i] for i in jb.lcut(text) ]
    # 转化为Torch接收的Tensor类型
    text2idx = torch.Tensor(text2idx).long()
    # 模型预测部分
    results = model(text2idx.view(-1,1))
    prediction = labels[torch.argmax(results,1).numpy()[0]]
    # --------------------------------------------------------------------------

    return prediction

sen = "不过这件事实在是真他妈的。而且她对我也起了疑心（这都是因为别人说我复杂），老是问：王二，你这人可靠吗？你能肯定自己没有偷过东西，或者趴过女厕所窗户吗？关于结婚的事，有一点开头我不明白。"

res = predict(sen)
print(res)