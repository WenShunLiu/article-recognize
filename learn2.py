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


embeds = nn.Embedding(100000, 5)
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(embeds(input).size())
st = "刘文顺是个大帅哥，赵璐婷是个臭猪猪"
print(jb.lcut(st))
