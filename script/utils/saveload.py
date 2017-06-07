import torch
import torch.nn as nn
import torch.nn.functional as F

def save(encoder,decoder,path):
    torch.save(model,path)

def save_para(model,path):
    torch.save(model.state_dict(), path)

def load(path):
    model = torch.load(path)
    return model

def load_para(path,modeltype):
    if modeltype == 'encoder':
        the_model = Encoder_ShowAndTellModel(*args, **kwargs)
    else# if modeltype == 'decoder':
        the_model = Decoder_ShowAndTellModel(*args, **kwargs)
    the_model.load_state_dict(torch.load(path))

