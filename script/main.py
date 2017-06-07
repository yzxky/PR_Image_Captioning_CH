#-*-coding: utf-8-*-

import random

import torch
torch.cuda.set_device(1)
import torch.nn as nn
from torch.autograd import Variable

from preprocessing.word2index.word2index import txt2list
from preprocessing.preparingData.input_data import input_data

from preprocessing.preparingData.prepare_data import *
from utils import global_variable
from utils.train import *
from utils.evaluate import *
from model.ShowAndTellModel import *
from model.ShowAndTellRevise import *

global_variable.train_set, global_variable.valid_set, global_variable.test_Set = input_data()
global_variable.train_set['lang'], global_variable.train_set['caption'] = txt2list('/data/PR_data/caption/train_single_process.txt')

input_size = 4096
hidden_size = 128
encoder1 = Encoder_ShowAndTellRevise(input_size, hidden_size)
decoder1 = Decoder_ShowAndTellRevise(hidden_size, global_variable.train_set['lang'].n_words, 1, drop_prob=0.1)

if global_variable.use_cuda:
    encoder1 = encoder1.cuda()
    decoder1 = decoder1.cuda()

for i in range(1000):
    trainIters_ShowAndTellRevise(encoder1, decoder1, 500, print_every=50, learning_rate=0.001)
    evaluateRandomly_ShowAndTellRevise(encoder1, decoder1)
    if (i + 1) % 10 is 0:
        torch.save(encoder1.state_dict(), 'saved_model/ST_encoder_hidden128_' + str(i+1) + '.pkl')
        torch.save(decoder1.state_dict(), 'saved_model/ST_decoder_hidden128' + str(i + 1) + '.pkl')

print "hello"
