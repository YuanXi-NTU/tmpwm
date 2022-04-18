import pickle
from torch.utils.data.dataset import  Dataset

class data(Dataset):
    def __init__(self,path):
        self.data = pickle.load(open(path, 'rb'))
        for k in list(self.data.keys()):
            self.data[k]=self.data[k].view(-1,self.data[k].shape[-1])
    def __getitem__(self, i):
        # data: 'rew_state', 'pos', 'vel', 'tau', -->obs
        #       'next_rew_state', 'next_pos', 'next_vel', 'next_tau',-->next_obs
        #       'action'

        # BUG IN DATASET, deprecated
        # return self.data['rew_state'][i][:7],self.data['pos'][i], self.data['vel'][i], \
        #        self.data['next_pos'][i],self.data['next_vel'][i],\
        #        self.data['tau'][i][:24],self.data['tau'][i][24:],self.data['next_tau'][i]
        return self.data['next_rew_state'][i][:7], self.data['next_pos'][i], self.data['next_vel'][i], \
        self.data['pos'][i], self.data['vel'][i], \
        self.data['next_tau'][i][:24], self.data['next_tau'][i][24:], self.data['tau'][i]


    def __len__(self):
        return self.data['pos'].shape[0]
