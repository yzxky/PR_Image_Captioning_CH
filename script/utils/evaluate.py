from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from preprocessing.preparingData.prepare_data import *
from utils import global_variable

SOS_token = 1
EOS_token = 2
use_cuda = global_variable.use_cuda

def evaluate(encoder, decoder, img_feats, lang, max_length):
    # TODO function variableFromImgFeat
    input_variable = variableFromFeature(img_feats)
    
    encoder_outputs = Variable(torch.zeros(encoder.output_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    encoder_outputs = encoder(input_variable)

    decoder_input   = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input   = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden  = decoder.initHiddenFromFeats(encoder_outputs)

    decoded_words   = []
    
    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
        topv, topi  = decoder_output.data.topk(1)
        ni          = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words

def evaluateRandomly(encoder, decoder, n = 5):
    for i in range(n):
        pair = randomChoosePairFromSet(global_variable.train_set, 'fc1')
        output_words = evaluate(encoder, decoder, pair[0], global_variable.train_set['lang'], 20)
        print(''.join(output_words))