import torch
import numpy as np
from numpy.typing import NDArray
import copy
import random

rng = np.random.default_rng()

class DataGenerator(torch.utils.data.Dataset):

    def __init__(self,
                 batch_size: int,
                 xp: NDArray,
                 xp_err: NDArray
        ):

        self.batch_size = batch_size

        # get rid of nans if they exist
        nan_mask = (np.sum(~np.isnan(xp),axis=-1).astype('bool'))*(np.sum(~np.isnan(xp_err),axis=-1).astype('bool'))
        self.xp = copy.deepcopy(xp)
        self.xp_err = copy.deepcopy(xp_err)

        self.data_length = len(self.xp)
        self.steps_per_epoch = self.data_length // self.batch_size
        self.idx_list = np.arange(self.data_length)

        self.epoch_xp = None
        self.epoch_xp_err = None

        self.epoch_end()

    def __iter__(self):
        for i in range(len(self)):
            tmp_idxs = self.idx_list[i*self.batch_size:(i+1)*self.batch_size]
            yield (self.epoch_xp[tmp_idxs],self.epoch_xp_err[tmp_idxs])

    def __len__(self):
        return self.steps_per_epoch
    
    def epoch_end(self):
        tmp = list(zip(self.xp,self.xp_err))
        tmp = np.random.permutation(self.data_length)
        self.epoch_xp,self.epoch_xp_err = self.xp[tmp],self.xp_err[tmp]
        
