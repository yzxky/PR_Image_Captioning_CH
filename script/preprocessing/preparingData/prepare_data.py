#-*-coding: utf-8-*-

import random

import torch
from torch.autograd import Variable

SOS_token = 1
EOS_token = 2
use_cuda = torch.cuda.is_available()

def variableFromFeature(feat):
    result = Variable(torch.Tensor(feat).view(-1, feat.size))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variableFromSentence(sentence):
    sentence.append(EOS_token)
    result = Variable(torch.LongTensor(sentence).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variableFromPair(pair):
    input_variable  = variableFromFeature(pair[0])
    target_variable = variableFromSentence(pair[1])
    return (input_variable, target_variable)


def randomChoosePair(feats_arr, cap_lst):
    index, caption = random.choice(cap_lst)
    return (feats_arr[index - 1], caption)


def randomChoosePairFromSet(set, img_key):
    return randomChoosePair(set[img_key], set['caption'])