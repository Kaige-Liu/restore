# -*- coding:utf-8 _*-
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def return_iter(args, split='train'):  # 对数据集进行抓牌
    data_eur = EurDataset(split)  # 得到train_data.pkl文件的内容（训练的数据集）
    # 下面对数据集进行抓牌，每次抓取batch_size个数据（句子）
    data_iter = DataLoader(data_eur, batch_size=args.batch_size, num_workers=0,
                           pin_memory=True, collate_fn=collate_data, shuffle=True, drop_last=True)  #
    return data_iter  # 返回一个迭代器dataloader,就是分好batch的数据集

def return_iter_10(args, split='train'):  # 对数据集进行抓牌
    data_eur = EurDataset_10(split)  # 得到train_data.pkl文件的内容（训练的数据集）
    # 下面对数据集进行抓牌，每次抓取batch_size个数据（句子）
    data_iter = DataLoader(data_eur, batch_size=args.batch_size, num_workers=0,
                           pin_memory=True, collate_fn=collate_data, shuffle=True, drop_last=True)  #
    return data_iter  # 返回一个迭代器dataloader,就是分好batch的数据集

def return_iter_eve(args, split='train'):  # eve的数据集
    data_eur = EurDataset_eve(split)  # 得到train_data.pkl文件的内容（训练的数据集）
    # 下面对数据集进行抓牌，每次抓取batch_size个数据（句子）
    data_iter = DataLoader(data_eur, batch_size=args.batch_size, num_workers=0,
                           pin_memory=True, collate_fn=collate_data, shuffle=True, drop_last=True)  #
    return data_iter  # 返回一个迭代器dataloader,就是分好batch的数据集


class EurDataset(Dataset):  # 数据集类，就是对pkl数据进行读取
    def __init__(self, split='train'):
        data_dir = './data/'
        with open(data_dir + '{}_data_alice.pkl'.format(split), 'rb') as f:  # 打开train_data.pkl文件
            self.data = pickle.load(f)  # 读取文件内容（类似于一本书，很多个句子，每个句子又有很多单词，单词是数字）


    def __getitem__(self, index):
        sents = self.data[index]  # 取出第index个句子,list
        return sents

    def __len__(self):
        return len(self.data)  # 返回这本书中句子的个数


class EurDataset_10(Dataset):  # 数据集类，就是对pkl数据进行读取
    def __init__(self, split='train'):
        data_dir = './data/'
        with open(data_dir + '{}_data_10.pkl'.format(split), 'rb') as f:  # 打开train_data.pkl文件
            self.data = pickle.load(f)  # 读取文件内容（类似于一本书，很多个句子，每个句子又有很多单词，单词是数字）


    def __getitem__(self, index):
        sents = self.data[index]  # 取出第index个句子,list
        return sents

    def __len__(self):
        return len(self.data)  # 返回这本书中句子的个数

class EurDataset_eve(Dataset):  # 数据集类，就是对pkl数据进行读取
    def __init__(self, split='train'):
        data_dir = './data/'
        with open(data_dir + '{}_data_eve.pkl'.format(split), 'rb') as f:  # 打开train_data.pkl文件
            self.data = pickle.load(f)  # 读取文件内容（类似于一本书，很多个句子，每个句子又有很多单词，单词是数字）


    def __getitem__(self, index):
        sents = self.data[index]  # 取出第index个句子,list
        return sents

    def __len__(self):
        return len(self.data)  # 返回这本书中句子的个数


def collate_data(batch):

    batch_size = len(batch)
    max_len = max(map(lambda x: len(x), batch))   # 当前batch的句子的最大长度
    max_len = 31
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent  # padding the questions

    return torch.from_numpy(sents)
