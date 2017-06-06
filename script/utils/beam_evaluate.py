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

def beam_evaluate(encoder, decoder, img_feats, lang, max_length,beam_num = 2):
    # TODO function variableFromImgFeat
    input_variable = variableFromImgFeat(img_feats)
    
    encoder_outputs = Variable(torch.zeros(encoder.output_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda() else encoder_outputs

    encoder_outputs = encoder(input_variable)

    decoder_input   = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input   = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden  = decoder.initHidden(encoder_outputs)

    decoded_words   = [[] for row in range(beam_num)]
    decoded_prob    = torch.Tensor([0  for row in range(beam_num)])
    beam_hidden     = torch.Tensor([torch.zeros(beam_num,decoder.rnn_hidden_size)])
    beam_output     = torch.Tensor([torch.zeros(beam_num,decoder.rnn_output_size)])
    next_word       = [[0,0] for row in range(beam_num)]

    for di in range(max_length):
        # Append beam_num words and put into beam_num list of words
        if len(decoded_words) == 0:
            decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi  = decoder_output.data.topk(beam_num)
            for wi in range(beam_num):
                ni = topi[0][wi]
                if ni == EOS_token:
                    decoded_words[wi].append('<EOS>')
                else:
                    decoded_words[wi].append(lang.index2word[ni])
                decoded_prob[wi] = topv[0][wi]
                beam_hidden[wi,:] = decoder_hidden
        # Get beam_num words for each word in list, and keep top beam_num choice
        else:
            for wi in range(beam_num):
                # TODO function to deal with EOS
                decoder_input = Varialbe(torch.LongTensor([[decoded_words[wi][-1]]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                decoder_output,beam_hidden[wi,:]  = decoder(
                        decoder_input, beam_hidden[wi,:])
                beam_output[wi,:] = decoder_output[0]
                for i in range(decoder_output.size()[1]):
                    beam_output[wi,i] += decoded_prob[wi]
            topv,topi = beam_output.view(1,-1).topk(beam_num)
            temp_words = decoded_words
            for wi in range(beam_num):
                row = topi[0][wi]/decoder.rnn_output_size
                ni = topi[0][wi]%decoder.rnn_output_size
                decoded_words[wi] = temp_words[row]
                decoded_prob[wi] = topv[0][wi]
                if ni == EOS_token:
                    decoded_words[wi].append('<EOS>')
                else:
                    decoded_words[wi].append(lang.index2word[ni])
    #choose best sentence as output
    i = decoded_prob.topk(1)
    decoded_sentence = decoded_words[i[0][0]]
    return decoded_sentense
