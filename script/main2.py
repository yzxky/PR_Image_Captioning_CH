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
from preprocessing.load_data import load_data

global_variable.train_set, global_variable.valid_set, global_variable.test_Set = input_data()
global_variable.train_set['lang'], global_variable.train_set['caption'] = txt2list('/data/PR_data/caption/train_single_process.txt')

####################################
#            DataLoader            #
####################################
BATCH_SIZE = 64
image_cap_dataset = load_data('train', sen_len=15)

loader = Data.DataLoader(
    dataset=image_cap_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=16
)

input_size = 4096
hidden_size = 256
encoder1 = Encoder_ShowAndTellModel(input_size, hidden_size)
decoder1 = Decoder_ShowAndTellModel(hidden_size, global_variable.train_set['lang'].n_words, 1, drop_prob=0.1)

learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder1.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder1.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

if global_variable.use_cuda:
    encoder1 = encoder1.cuda()
    decoder1 = decoder1.cuda()


for epoch in range(100):
    for step, (batch_x, batch_y) in enumerate(loader):
        print_loss_total = 0
        input_variable = Variable(batch_x)  #.view(-1, 1, input_size)
        target_variable = Variable(batch_y) #.view(-1, 1, hidden_size)
        input_variable = input_variable.cuda() if global_variable.use_cuda else input_variable
        target_variable = target_variable.cuda() if global_variable.use_cuda else target_variable

        loss = train(input_variable, target_variable, encoder1,
                     decoder1, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        print('Epoch:', epoch, '|Step:', step, '|loss:', print_loss_total)
        if step % 50 == 0:
            evaluateRandomly(encoder1, decoder1)

    torch.save(encoder1.state_dict(), 'saved_model/ST_encoder_epoch_' + str(epoch + 1) + '.pkl')
    torch.save(decoder1.state_dict(), 'saved_model/ST_decoder_epoch_' + str(epoch + 1) + '.pkl')


    #for i in range(1000):
#    trainIters(encoder1, decoder1, 500, print_every=50, learning_rate=0.001)
#    evaluateRandomly(encoder1, decoder1)
#    if (i + 1) % 10 is 0:

print "hello"
