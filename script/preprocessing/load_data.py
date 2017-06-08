#-*-coding: utf-8-*-

import torch
import torch.utils.data as Data
from preprocessing.word2index.word2index import txt2list
from preprocessing.preparingData.input_data import input_data
import numpy as np
from utils.evaluate import *
from model.ShowAndTellRevise import *

SOS_token = 1
EOS_token = 2

def load_data(data_type, data_name, sen_len=15):
    train_set, valid_set, test_Set = input_data(
        flag_fc1=True, flag_fc2=True, flag_pool=True)
    train_set['lang'], train_set['caption'] = txt2list(
        '/data/PR_data/caption/train_single_process.txt', sen_len)
    valid_set['lang'], valid_set['caption'] = txt2list(
        '/data/PR_data/caption/valid_single_process.txt', sen_len)

    if data_type is 'train':
        data = []
        target = []
        for cap in train_set['caption']:
            data.append(train_set[data_name][cap[0] - 1])
            cap[1].append(EOS_token)
            target.append(cap[1])
        dataTensor = torch.Tensor(np.array(data))
        targetTensor = torch.LongTensor(np.array(target))
    elif data_type is 'valid':
        data = []
        target = []
        for cap in valid_set['caption']:
            data.append(valid_set[data_name][cap[0] - 1])
            target.append(cap[1].append(EOS_token))
        dataTensor = torch.Tensor(np.array(data))
        targetTensor = torch.LongTensor(np.array(target))

    dataset = Data.TensorDataset(data_tensor=dataTensor, target_tensor=targetTensor)
    return dataset