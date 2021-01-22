import os
import torch

def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = [] # 片段
    target = [] # 作者
    
    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file) and file[0] != '.':
            f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index,line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    return list(zip(sentences, target))

def datasplit(dataset, tsize):
    ts = int(len(dataset)*tsize)
    vs = len(dataset) - ts
    train_d, valid_d = torch.utils.data.random_split(dataset, [ts, vs])
    return train_d, valid_d

if __name__ == "__main__":
    path = "dataset/"
    res = load_data(path)
    print(res[:3])