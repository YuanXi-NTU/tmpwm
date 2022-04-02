""" Some data loading utilities """
from bisect import bisect
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
import torch
import numpy as np
import pickle
from torch.utils.data.dataset import  Dataset

class RolloutSequenceDataset(Dataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.

    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of tuples (obs, action, reward, terminal, next_obs):
    - obs: (seq_len, *obs_shape)
    - actions: (seq_len, action_size)
    - reward: (seq_len,)
    - terminal: (seq_len,) boolean
    - next_obs: (seq_len, *obs_shape)

    NOTE: seq_len < rollout_len in moste use cases

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """
    def __init__(self, path, seq_len,  train=True): # pylint: disable=too-many-arguments
        self.buffer=pickle.load(open(path,'rb'))
        self.buffer={key:self.buffer[key].transpose(0,1) for key in list(self.buffer.keys())}#1000,4096,_->4096,1000,_
        if train:
            self.buffer={key:self.buffer[key][:int(0.8*self.buffer[key].shape[0])] for key in list(self.buffer.keys())}
        else:
            self.buffer={key:self.buffer[key][int(0.8*self.buffer[key].shape[0]):] for key in list(self.buffer.keys())}
        self.seq_len = seq_len
        self.train=train
    def _get_data(self, data, seq_index):
        obs_data = data['observations'][seq_index:seq_index + self._seq_len + 1]
        obs_data = self._transform(obs_data.astype(np.float32))
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][seq_index+1:seq_index + self._seq_len + 1]
        action = action.astype(np.float32)
        reward, terminal = [data[key][seq_index+1:
                                      seq_index + self._seq_len + 1].astype(np.float32)
                            for key in ('rewards', 'terminals')]
        # data is given in the form
        # (obs, action, reward, terminal, next_obs)
        return obs, action, reward, terminal, next_obs

    def __len__(self):
        return (self.buffer['obs'].shape[1]-self.seq_len+1)*self.seq_len
    def __getitem__(self, i):
        idx1,idx2=i//self.buffer['obs'].shape[1],i%self.buffer['obs'].shape[1]
        return self.buffer['obs'][idx1,idx2:idx2+self.seq_len,:],\
                self.buffer['action'][idx1,idx2:idx2+self.seq_len,:],\
                self.buffer['reward'][idx1,idx2:idx2+self.seq_len],\
                self.buffer['done'][idx1,idx2:idx2+self.seq_len],\
                self.buffer['next_obs'][idx1,idx2:idx2+self.seq_len,:]


class RolloutObservationDataset(Dataset): # pylint: disable=too-few-public-methods
    """ trajectories with {obs,reward,action,next_obs,done}

    :args path: path directory of data sequences
    :args train: if True, train data, else test
    """
    def __init__(self, path=None, train=True):
        self.buffer=pickle.load(open(path,'rb'))
        self.buffer['obs']=self.buffer['obs'].view(-1,60)
        self.buffer['next_obs']=self.buffer['next_obs'].view(-1,60)
        self.buffer['action']=self.buffer['action'].view(-1,8)
        # self.buffer['reward']=self.buffer.view(-1)
        #self.buffer['done']=self.buffer.view(-1)

        split_pos = int(0.8 * self.buffer['obs'].shape[0])
        if train:
            self.buffer['obs'] = self.buffer['obs'][:split_pos]
            self.buffer['next_obs'] = self.buffer['next_obs'][:split_pos]
            self.buffer['action'] = self.buffer['action'][:split_pos]
        else:
            self.buffer['obs'] = self.buffer['obs'][split_pos:]
            self.buffer['next_obs'] = self.buffer['next_obs'][split_pos:]
            self.buffer['action'] = self.buffer['action'][split_pos:]

    def __getitem__(self, i):
        return self.buffer['obs'][i],self.buffer['action'][i],self.buffer['next_obs'][i]
    def __len__(self):
        return self.buffer['action'].shape[0]
