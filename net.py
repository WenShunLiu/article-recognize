import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num):
        super(Net,self).__init__()
        # embedding层，生成词向量
        self.embeds = nn.Embedding(num, 80)
        # lstm层，1层结构
        self.lstm = nn.LSTM(80, 40)
        self.lf1 = nn.Linear(40, 64)
        self.lf2 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.embeds(x)
        x, hidden = self.lstm(x)
        lastx = hidden[0][0] # h_n隐藏层最后一层的输出
        lastx = F.tanh(lastx)
        lastx = self.lf1(lastx)
        lastx = F.leaky_relu(lastx)
        lastx = self.lf2(lastx)
        return lastx

if __name__ == "__main__":
    net = Net(1000)
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
        print(parameters)