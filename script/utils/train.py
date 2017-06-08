from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from preprocessing.preparingData.prepare_data import variableFromPair
from preprocessing.preparingData.prepare_data import randomChoosePairFromSet
from utils import global_variable

use_cuda = global_variable.use_cuda

teacher_forcing_ratio = 1

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size      = target_variable.size()[0]
    target_length   = target_variable.size()[1]

    encoder_outputs = Variable(torch.zeros(encoder.output_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    encoder_outputs = encoder(input_variable)

    decoder_input   = Variable(torch.LongTensor(
        [[global_variable.SOS_token] for _ in range(batch_size)]))
    decoder_input   = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden  = decoder.initHiddenFromFeats(encoder_outputs)
    
#    decoder_output, decoder_hidden = decoder(
#            encoder_outputs, decoder_hidden)
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            test_a = decoder_output[:, 0]
            test_b = target_variable[:, di]

            loss += criterion(decoder_output, target_variable[:, di])
            decoder_input = target_variable[:, di]

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
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

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    print_loss_total = 0

    encoder_optimizer   = optim.Adam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer   = optim.Adam(decoder.parameters(), lr = learning_rate)

    training_pairs      = [variableFromPair(randomChoosePairFromSet(global_variable.train_set, 'fc1'))
                           for i in range(n_iters)]
    criterion           = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair   = training_pairs[iter - 1]
        input_variable  = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder, 
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(print_loss_avg)


