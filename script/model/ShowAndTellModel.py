from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ShowAndTellModel(nn.Module):
    def __init__(self, settings):
        super(ShowAndTellModel, self).__init__()
        
        # Model Parameter
        self.fc_feat_size       = settings.fc_feat_size
        

        # RNN Parameter
        self.rnn_input_size     = settings.rnn_input_size
        self.rnn_hidden_size    = settings.rnn_hidden_size
        self.rnn_num_layers     = settings.rnn_num_layers
        self.rnn_drop_prob      = settings.rnn_drop_prob

        
        self.embed  = nn.Embedding(self.vocab_size, self.rnn_input_size)
        self.rnn    = nn.LSTM(self.rnn_input_size, self.rnn_hidden_size, self.rnn_num_layers, dropout=self.rnn_drop_prob)
        self.out    = nn.Linear(self.rnn_hidden_size, self.rnn_input_size)
        self.softmax= nn.LogSoftmax()


