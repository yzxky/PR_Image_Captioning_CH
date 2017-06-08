#-*-coding: utf-8-*-

import random
import torch
torch.cuda.set_device(1)
import torch.nn as nn
from torch.autograd import Variable

from utils import global_variable
from utils.evaluate import evaluate
from utils.beam_evaluate import beam_evaluate
from preprocessing.preparingData.input_data import input_data
from preprocessing.word2index.word2index import txt2list
from utils.saveload import load_para
from utils.criteria import bleu_score
from model.ShowAndTellModel import *
from model.ShowAndTellRevise import *
from model.ST_ImageExtended import *

def bleu_test(global_variable,encoder,decoder,maxLabel,mode = 'top1',feature = 'fc1',maxLength = 20,ngram = 1):
    # evaluate
    reference = [[] for i in range(5)]
    file = open("/data/PR_data/caption/valid_single.txt")
    ave_score = 0
    for label in range(8001,maxLabel+1):
        line = file.readline()
        for i in range(5):
            line = file.readline().decode("utf-8")
            reference[i] = line.split(' ')
            if mode == 'top1':
                predict = evaluate_ST_ImageExtended(encoder, decoder, global_variable.valid_set[feature][label-8001],
                        global_variable.train_set['lang'], maxLength)
                #if predict[-1] is u'<EOS>':
                predict[-1] = ''
                predict_sen = ' '.join(predict)
                predict = predict_sen.split()

            else:
                predict = beam_evaluate(encoder, decoder, 
                        global_variable.valid_set[feature][label-8001],
                        global_variable.train_set['lang'],maxLength)
        score = bleu_score(reference,predict)
        ave_score = ave_score + score
        print('label:%d BLEU-1:%f' % (label,score))
        print(' '.join(predict))
        print(' '.join(reference[0]))

    # finish
    ave_score = ave_score / (float)(label-8000)
    print('finish, average_bleu-1: %f' % ave_score)
    return ave_score
