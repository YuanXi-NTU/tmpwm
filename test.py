#test for sim env, basic parts for vecenv

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask
import easydict,yaml
from models.vae import VAE
from models.mdrnn import MDRNNCell
from torch.distributions.categorical import Categorical
class Test(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg
        self.sim = None #in create_sim

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        # self.randomization_params = self.cfg["task"]["randomization_params"]
        # self.randomize = self.cfg["task"]["randomize"]


        self.cfg["env"]["numObservations"] = 60
        self.cfg["env"]["numActions"] = 8

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)


        self.args = easydict.EasyDict(yaml.load(open('./env_config.yaml'), yaml.FullLoader))
        self.vae = VAE(self.args.obs_shape, self.args.obs_shape, self.args.vae_latent_size)
        self.mdrnn = MDRNNCell(self.args.model.latent_size,
                               self.args.action_shape, self.args.model.rnn_size,
                               self.args.model.num_mixtures)
        vae_path = '/home/yuanxi20/tmpwm/vae.pth'
        mdrnn_path = '/home/yuanxi20/tmpwm/mdrnn.pth'
        self.vae.load_state_dict(torch.load(vae_path)['vae'])
        self.mdrnn.load_state_dict(torch.load(mdrnn_path)['mdrnn'])
        self.vae.eval()
        self.mdrnn.eval()

    def reset_idx(self, env_ids):

        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

        # positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        # velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
        #                                      self.dof_limits_upper)
        # self.dof_vel[env_ids] = velocities

        '''
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        '''

        # to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        # to_target[:, 2] = 0.0
        # self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        # self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self._lstate[env_ids]=torch.randn(len(env_ids),self.args.vae_latent_size)
        self._hstate[env_ids]=2*torch.randn(len(env_ids),self.args.vae_latent_size)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        #reset obs


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
        #### ???
        reset = torch.where(self.obs_buf[:, 0] < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)
        return reset


    def step(self, actions: torch.Tensor) :#-> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        with torch.no_grad():
            action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
            # apply actions
            self.pre_physics_step(action_tensor)
            ####
            # actions = torch.Tensor(action).unsqueeze(0)
            mu, sigma, pi, self.rew_buf, self.reset_buf, n_h = self.mdrnn(actions, self._lstate, self._hstate)
            pi = pi.squeeze()
            mixt = Categorical(torch.exp(pi)).sample().item()

            self._lstate = mu[:, mixt, :] + sigma[:, mixt, :] * \
                           torch.randn_like(mu[:, mixt, :])
            self._hstate = n_h

            self.obs_dict["obs"] = self.vae.decoder(self._lstate)
        ####

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