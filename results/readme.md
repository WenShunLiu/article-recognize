V1:
```python
class Net(nn.Module):
    def __init__(self, num):
        super(Net,self).__init__()
        # embedding层，生成词向量
        self.embeds = nn.Embedding(num, 32)
        # lstm层，1层结构
        self.lstm = nn.LSTM(32, 16)
        self.lf1 = nn.Linear(16, 64)
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

range = 10
```

v2:
```python
class Net(nn.Module):
    def __init__(self, num):
        super(Net,self).__init__()
        # embedding层，生成词向量
        self.embeds = nn.Embedding(num, 32)
        # lstm层，1层结构
        self.lstm = nn.LSTM(32, 16)
        self.lf1 = nn.Linear(16, 64)
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

range = 40
```

v3
```python
class Net(nn.Module):
    def __init__(self, num):
        super(Net,self).__init__()
        # embedding层，生成词向量
        self.embeds = nn.Embedding(num, 72)
        # lstm层，1层结构
        self.lstm = nn.LSTM(72, 32)
        self.lf1 = nn.Linear(32, 64)
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

range = 20
```
v4
```python
class Net(nn.Module):
    def __init__(self, num):
        super(Net,self).__init__()
        # embedding层，生成词向量
        self.embeds = nn.Embedding(num, 72)
        # lstm层，1层结构
        self.lstm = nn.LSTM(72, 32)
        self.lf1 = nn.Linear(32, 64)
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

range = 40
```

v5:
```python
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

range = 40
```