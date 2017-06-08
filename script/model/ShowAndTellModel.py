from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import global_variable

use_cuda = global_variable.use_cuda

class Encoder_ShowAndTellModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder_ShowAndTellModel, self).__init__()

        self.input_size         = input_size
        self.output_size        = output_size

        self.img_embedding      = nn.Linear(self.input_size, self.output_size)

    def forward(self, input):
        output = self.img_embedding(input)
        return output
        

class Decoder_ShowAndTellModel(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers = 1, drop_prob=0.1):
        super(Decoder_ShowAndTellModel, self).__init__()
        
        # RNN Parameter
        self.rnn_hidden_size    = hidden_size
        self.rnn_output_size    = output_size
        self.rnn_num_layers     = num_layers
        self.rnn_drop_prob      = drop_prob

        self.embedding  = nn.Embedding(self.rnn_output_size, self.rnn_hidden_size)
        self.rnn        = nn.LSTM(self.rnn_hidden_size, self.rnn_hidden_size, self.rnn_num_layers, dropout=self.rnn_drop_prob)
        self.out        = nn.Linear(self.rnn_hidden_size, self.rnn_output_size)
        self.softmax    = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = output.view(1, -1, self.rnn_hidden_size)
        for i in range(self.rnn_num_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.rnn_hidden_size))
        c0 = Variable(torch.zeros(1, batch_size, self.rnn_hidden_size))
#        result = (Variable(torch.zeros(1, 1, self.rnn_hidden_size)),
#                  Variable(torch.zeros(1, 1, self.rnn_hidden_size)))
        if use_cuda:
            return (h0.cuda(), c0.cuda())
        else:
            return (h0, c0)

    def initHiddenFromFeats(self, img_feats):
        hidden = self.initHidden(img_feats.size()[0])
        for i in range(self.rnn_num_layers):
            output = F.relu(img_feats).view(1, -1, self.rnn_hidden_size)
            output, hidden = self.rnn(output, hidden)
        return hidden


teacher_forcing_ratio = 1
