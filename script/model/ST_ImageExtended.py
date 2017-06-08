from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import global_variable
from preprocessing.preparingData.prepare_data import *

use_cuda = global_variable.use_cuda
teacher_forcing_ratio = 1

class Encoder_ST_ImageExtended(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder_ST_ImageExtended, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.img_embedding = nn.Linear(self.input_size, self.output_size)

    def forward(self, input):
        output = self.img_embedding(input)
        return output


class Decoder_ST_ImageExtended(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, drop_prob=0.1):
        super(Decoder_ST_ImageExtended, self).__init__()

        # RNN Parameter
        self.rnn_hidden_size = hidden_size
        self.rnn_output_size = output_size
        self.rnn_num_layers = num_layers
        self.rnn_drop_prob = drop_prob

        self.embedding = nn.Embedding(self.rnn_output_size,
                                      self.rnn_hidden_size)
        self.rnn = nn.LSTM(2 * self.rnn_hidden_size,
                           self.rnn_hidden_size,
                           self.rnn_num_layers,
                           dropout=self.rnn_drop_prob)
        self.out = nn.Linear(self.rnn_hidden_size,
                             self.rnn_output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, feats, input, hidden):
        output = self.embedding(input).view(-1, 1, self.rnn_hidden_size)
        feats = feats.view(-1, 1, self.rnn_hidden_size)
        output = torch.cat([feats, output], 2)
        output = output.view(1, -1, 2 * self.rnn_hidden_size)
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
        output = torch.cat([img_feats, img_feats], 1)
        output = F.relu(output).view(1, -1, 2 * self.rnn_hidden_size)
        for i in range(self.rnn_num_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)
        return hidden


def train_ST_ImageExtended(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size = target_variable.size()[0]
    target_length = target_variable.size()[1]

    encoder_outputs = Variable(torch.zeros(encoder.output_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    encoder_outputs = encoder(input_variable)

    decoder_input = Variable(torch.LongTensor(
        [[global_variable.SOS_token] for _ in range(batch_size)]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    #decoder_hidden = decoder.initHidden()
    decoder_hidden = decoder.initHiddenFromFeats(encoder_outputs)

    #    decoder_output, decoder_hidden = decoder(
    #            encoder_outputs, decoder_hidden)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                encoder_outputs, decoder_input, decoder_hidden)

            test_a = decoder_output[:, 0]
            test_b = target_variable[:, di]

            loss += criterion(decoder_output, target_variable[:, di])
            decoder_input = target_variable[:, di]

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                encoder_outputs, decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output[0], target_variable[di])
            if ni == global_variable.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def evaluate_ST_ImageExtended(encoder, decoder, img_feats, lang, max_length):
    # TODO function variableFromImgFeat
    input_variable = variableFromFeature(img_feats)

    encoder_outputs = Variable(torch.zeros(encoder.output_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    encoder_outputs = encoder(input_variable)

    decoder_input = Variable(torch.LongTensor([[global_variable.SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = decoder.initHiddenFromFeats(encoder_outputs)

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            encoder_outputs, decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words


def evaluateRandomly_ST_ImageExtended(encoder, decoder, data_name, n=5):
    for i in range(n):
        pair = randomChoosePairFromSet(global_variable.train_set, data_name)
        output_words = evaluate_ST_ImageExtended(encoder, decoder, pair[0], global_variable.train_set['lang'], 20)
        print(''.join(output_words))