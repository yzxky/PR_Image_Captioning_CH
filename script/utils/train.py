from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

SOS_token = 1
EOS_token = 2

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length   = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(encoder.output_size))
    encoder_outputs = encoder_outputs.cuda() if use_cude else encoder_outputs

    loss = 0

    encoder_outputs = encoder(input_variable)

    decoder_input   = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input   = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden  = decoder.initHidden(encoder_outputs)
    
#    decoder_output, decoder_hidden = decoder(
#            encoder_outputs, decoder_hidden)
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di]

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output[0], target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    
    encoder_optimizer   = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer   = optim.SGD(decoder.parameters(), lr = learning_rate)
    # TODO function: variablesFromPair
    training_pairs      = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]
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


