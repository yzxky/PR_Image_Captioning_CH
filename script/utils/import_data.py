#-*-coding=utf-8-*-
import h5py
import numpy as np
import torch

def import_data(flag_fc1 = True, flag_fc2 = True, flag_pool = False):
    # Import data of image features 
    fc1 = {}
   
    if flag_fc1 is True:
        f_fc1 = h5py.File('/data/image_vgg19_fc1_feature.h5', 'r')
        # import data
        fc1['train_set'] = torch.Tensor(np.transpose(f_fc1['train_set']))
        fc1['validation_set'] = torch.Tensor(np.tranpose(f_fc1['validation_set']))
        fc1['test_set']  = torch.Tensor(np.transpose(f_fc1['test_set']))
