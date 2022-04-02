from models.vae import VAE
from models.mdrnn import MDRNNCell
from torch.distributions.categorical import Categorical
import easydict
import yaml
import gym
import torch
import os

import numpy as np
from gym import spaces


class CusEnv(gym.Env):
    def __init__(self):
        self.args = easydict.EasyDict(yaml.load(open('./env_config.yaml'), yaml.FullLoader))
        self.action_space=spaces.Box(np.ones(8),-np.ones(8))
        self.vae=VAE(self.args.obs_shape,self.args.obs_shape,self.args.vae_latent_size)
        self.mdrnn=MDRNNCell(self.args.model.latent_size,
                             self.args.action_shape,self.args.model.rnn_size,
                             self.args.model.num_mixtures)
        self.vae.load_state_dict()

    def reset(self):
        self._lstate = torch.randn(1, self.args.vae_latent_size)
        self._hstate = 2 * [torch.zeros(1, self.model.rnn_size)]
    def step(self, action):
        with torch.no_grad():
            action = torch.Tensor(action).unsqueeze(0)
            mu, sigma, pi, reward, done, n_h = self.mdrnn(action, self._lstate, self._hstate)
            pi = pi.squeeze()
            mixt = Categorical(torch.exp(pi)).sample().item()

            self._lstate = mu[:, mixt, :] + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
            self._hstate = n_h

            self._obs = self.vae.decoder(self._lstate)

            np_obs = self._obs.numpy()
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze()

            return np_obs, reward.item(), done.item() > 0

