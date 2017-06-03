#-*-coding=utf-8-*-
import h5py
import numpy as np
import torch

def input_data(flag_fc1 = True, flag_fc2 = True, flag_pool = False):
    # Import data of image features 
    train_set       = {}
    validation_set  = {}
    test_set        = {}
   
    if flag_fc1 is True:
        f_fc1 = h5py.File('/data/PR_data/image_vgg19_fc1_feature.h5', 'r')
        # import data into Tensor form
        train_set['fc1']        = torch.Tensor(np.transpose(f_fc1['train_set']))
        validation_set['fc1']   = torch.Tensor(np.transpose(f_fc1['validation_set']))
        test_set['fc1']         = torch.Tensor(np.transpose(f_fc1['test_set']))

    if flag_fc2 is True:
        f_fc2 = h5py.File('/data/PR_data/image_vgg19_fc2_feature.h5', 'r')
        # import data into Tensor form
        train_set['fc2']        = torch.Tensor(np.transpose(f_fc2['train_set']))
        validation_set['fc2']   = torch.Tensor(np.transpose(f_fc2['validation_set']))
        test_set['fc2']         = torch.Tensor(np.transpose(f_fc2['test_set']))

    if flag_pool is True:
        f_pool = h5py.File('/data/PR_data/image_vgg19_block5_pool_feature.h5', 'r')
        train_set['pool']       = torch.Tensor(np.transpose(f_pool['train_set']))
        validation_set['pool']  = torch.Tensor(np.transpose(f_pool['validation_set']))
        test_set['pool']        = torch.Tensor(np.transpose(f_pool['test_set']))

    return train_set, validation_set, test_set
