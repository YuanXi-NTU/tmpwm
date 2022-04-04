#code for dream env
import os
import torch
import easydict
import yaml

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

from .vae import VAE
from .mdrnn import MDRNNCell
from torch.distributions.categorical import Categorical

class Test(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.sim = None #in create_sim

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        # self.randomization_params = self.cfg["task"]["randomization_params"]
        # self.randomize = self.cfg["task"]["randomize"]


        self.cfg["env"]["numObservations"] = 60
        self.cfg["env"]["numActions"] = 8

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)

        # params usage
        self.args = easydict.EasyDict(yaml.load(open('/home/yuanxi20/tmpwm/wm_config.yaml'), yaml.FullLoader))

        self.vae_hidden_size = self.args.model.vae.hidden_size
        self.rnn_size = self.args.model.rnn.rnn_size
        self.rnn_latent_size = self.args.model.rnn.latent_size
        self.num_mixtures = self.args.model.rnn.num_mixtures

        self.vae = VAE(self.args.obs_shape, self.args.obs_shape, self.vae_hidden_size)
        self.mdrnn = MDRNNCell(self.rnn_latent_size,
                               self.args.action_shape, self.rnn_size,
                               self.num_mixtures)

        # refer envs/simualted carracing.py
        
        
        self._lstate = torch.randn(4096, self.vae_hidden_size).to(self.device)
        self._hstate = 2 * [torch.zeros(4096, self.rnn_size).to(self.device)]

        vae_path = self.args.sim_env.vae_path
        mdrnn_path = self.args.sim_env.mdrnn_path
        mdrnn_weight = torch.load(mdrnn_path)['mdrnn']

        mdrnn_weight['rnn.weight_ih']=mdrnn_weight['rnn.weight_ih_l0']
        mdrnn_weight['rnn.weight_hh']=mdrnn_weight['rnn.weight_hh_l0']
        mdrnn_weight['rnn.bias_ih']=mdrnn_weight['rnn.bias_ih_l0']
        mdrnn_weight['rnn.bias_hh']=mdrnn_weight['rnn.bias_hh_l0']
        del mdrnn_weight['rnn.weight_ih_l0'],mdrnn_weight['rnn.weight_hh_l0']
        del mdrnn_weight['rnn.bias_ih_l0'],mdrnn_weight['rnn.bias_hh_l0']


        self.mdrnn.load_state_dict(mdrnn_weight)
        self.vae.load_state_dict(torch.load(vae_path)['vae'])
        self.vae.to(self.device)
        self.mdrnn.to(self.device)
        self.vae.eval()
        self.mdrnn.eval()

    def reset_idx(self, env_ids):
        '''reset states of done envs'''

        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)


        # refer envs/simulated_carracing.py
        self._lstate[env_ids]=torch.randn(len(env_ids),self.vae_hidden_size).to(self.device)

        self._hstate[0][env_ids] = torch.randn(len(env_ids), self.rnn_size).to(self.device)
        self._hstate[1][env_ids] = torch.randn(len(env_ids), self.rnn_size).to(self.device)

        #reset obs
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # completed in step
        # self.compute_observations() #get self.obs_buf[]
        # self.compute_reward(self.actions)#get self.rew_buf[:], self.reset_buf[:]

        # # reset agents
        reset = torch.where(self.obs_buf[:, 0] < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)
        return reset


    def step(self, actions: torch.Tensor) :#-> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        with torch.no_grad():
            action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
            # apply actions
            self.pre_physics_step(action_tensor)

            mu, sigma, pi, self.rew_buf, self.reset_buf, n_h = self.mdrnn(actions, self._lstate, self._hstate)
            pi = pi.squeeze()
            mixt = Categorical(torch.exp(pi)).sample()#.item()
            dim_0_idx=torch.arange(mixt.shape[0])
            # self._lstate = mu[:, mixt, :] + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :]) #original, used in single envs
            self._lstate = mu[dim_0_idx, mixt, :] + sigma[dim_0_idx, mixt, :] * torch.randn_like(mu[dim_0_idx, mixt, :])
            self._hstate = n_h

            self.obs_dict["obs"] = self.vae.decoder(self._lstate)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # fill time out buffer
        self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1,
                                       torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))

        # compute observations, rewards, resets, ...
        self.reset_buf=self.post_physics_step()

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
        return self.obs_dict, self.rew_buf.to(self.rl_device), \
               self.reset_buf.to(self.rl_device), self.extras

    def create_sim(self):
        return

    def _create_ground_plane(self):
        return


    def _create_envs(self, num_envs, spacing, num_per_row):
        return
