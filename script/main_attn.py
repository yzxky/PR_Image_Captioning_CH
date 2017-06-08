#-*-coding: utf-8-*-

import random

import torch
torch.cuda.set_device(1)
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

from preprocessing.word2index.word2index import txt2list
from preprocessing.preparingData.input_data import input_data

from preprocessing.preparingData.prepare_data import *
from utils import global_variable
from utils.train import *
from utils.evaluate import *
from model.ShowAndTellModel import *
from model.ShowAndTellRevise import *
from model.ST_ImageExtended import *
from model.AttnModel import *
from preprocessing.load_data import load_data

global_variable.train_set, global_variable.valid_set, global_variable.test_Set = \
    input_data(flag_fc1=True, flag_fc2=True, flag_pool=True)
global_variable.train_set['lang'], global_variable.train_set['caption'] = txt2list('/data/PR_data/caption/train_single_process.txt')

####################################
#            DataLoader            #
####################################
BATCH_SIZE = 64
SEN_LEN = 15
image_cap_dataset = load_data('train', 'pool', sen_len=SEN_LEN)

loader = Data.DataLoader(
    dataset=image_cap_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=16
)

input_size = 512
hidden_size = 256
encoder1 = Encoder_Attn(feat_num=49,
                        input_size=input_size,
                        output_size=hidden_size)
decoder1 = Decoder_Attn(feat_num=49,
                        hidden_size=hidden_size,
                        output_size=global_variable.train_set['lang'].n_words,
                        num_layers=1,
                        drop_prob=0.1)

criterion = nn.NLLLoss()

if global_variable.use_cuda:
    encoder1 = encoder1.cuda()
    decoder1 = decoder1.cuda()


for epoch in range(100):
    learning_rate = 0.001 * 0.1 ** (epoch / 20)
    encoder_optimizer = optim.Adam(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder1.parameters(), lr=learning_rate)
    for step, (batch_x, batch_y) in enumerate(loader):
        print_loss_total = 0
        input_variable = Variable(batch_x)  #.view(-1, 1, input_size)
        target_variable = Variable(batch_y) #.view(-1, 1, hidden_size)
        input_variable = input_variable.cuda() if global_variable.use_cuda else input_variable
        target_variable = target_variable.cuda() if global_variable.use_cuda else target_variable

        loss = train_Attn(input_variable, target_variable, encoder1,
                     decoder1, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        print('Epoch:', epoch, '|Step:', step, '|loss:', print_loss_total)
        if step % 50 == 0:
            evaluateRandomly_Attn(encoder1, decoder1, 'pool')

    torch.save(encoder1.state_dict(), 'saved_model/Attn_encoder_epoch_decending_lr_' + str(epoch + 1) + '.pkl')
    torch.save(decoder1.state_dict(), 'saved_model/Attn_decoder_epoch_decending_lr_' + str(epoch + 1) + '.pkl')


    #for i in range(1000):
#    trainIters(encoder1, decoder1, 500, print_every=50, learning_rate=0.001)
#    evaluateRandomly(encoder1, decoder1)
#    if (i + 1) % 10 is 0:

print "hello"
