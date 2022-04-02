from models.vae import VAE
from models.mdrnn import MDRNNCell
import easydict
import yaml
import gym
import torch
import os

import numpy as np
from gym import spaces


class CusEnv(gym.Env):
    def __init__(self):
        args = easydict.EasyDict(yaml.load(open('./rnn_config.yaml'), yaml.FullLoader))
        self.action_space=spaces.Box(np.ones(8),-np.ones(8))
        self.vae=VAE(args.obs_shape,args.obs_shape,args.vae.latent_size)
        # self.mdrnn=MDRNNCell()
        self.vae.load_state_dict()

    def reset(self):
        self._lstate = torch.randn(1, LSIZE)
        self._hstate = 2 * [torch.zeros(1, RSIZE)]
    def step(self, action):
        with torch.no_grad():
            action = torch.Tensor(action).unsqueeze(0)
            mu, sigma, pi, reward, done, n_h = self._rnn(action, self._lstate, self._hstate)
            pi = pi.squeeze()
            mixt = Categorical(torch.exp(pi)).sample().item()

            self._lstate = mu[:, mixt, :] # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
            self._hstate = n_h

            self._obs = self._decoder(self._lstate)
            np_obs = self._obs.numpy()
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze()
            np_obs = np_obs.astype(np.uint8)
            self._visual_obs = np_obs

            return np_obs, reward.item(), done.item() > 0

