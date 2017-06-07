#-*-coding=utf-8-*-
import torch

SOS_token = 1
EOS_token = 2
use_cuda = torch.cuda.is_available()
train_set = {}
valid_set = {}
test_set = {}
