import easydict,yaml,pickle
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader
from models import MDRNN, gmm_loss, VAE

args=easydict.EasyDict(yaml.load(open('./rnn_config.yaml'),yaml.FullLoader))

model = MDRNN(args.vae_latent_size, args.action_shape, args.latent_size, args.num_mixtures).cuda()
vae= VAE(args.vae_latent_size).cuda()
vae_path='./vae.pth'
vae.load_state_dict(torch.load(vae_path)['vae'])

path = '/home/yuanxi20/isaacgym/IsaacGymEnvs/isaacgymenvs/buffer_data/res_buffer.pickle'
buffer = pickle.load(open(path, 'rb'))
class SeqData(Dataset):
    def __init__(self, mu, logvar, actions, rewards, dones):
        seq_length = args.model.rnn_seq_len
        total_frames = mu.shape[0]
        num_batches = total_frames // seq_length
        N = num_batches * seq_length

        # self.mu = mu[:N].reshape(-1, seq_length, args.vae_z_size)
        # self.logvar = logvar[:N].reshape(-1, seq_length, args.vae_z_size)
        self.mu,self.logvar=vae.encoder(self.obs)
        self.obs= obs[:N].reshape(-1, seq_length, args.obs_shape)
        self.actions = actions[:N].reshape(-1, seq_length)
        self.rewards = rewards[:N].reshape(-1, seq_length)
        self.dones = dones[:N].reshape(-1, seq_length)

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, idx):
        return self.mu[idx], self.logvar[idx], self.actions[idx], \
                self.rewards[idx], self.dones[idx]
buffer=SeqData(buffer['obs'],buffer['action'],buffer['reward'],buffer['done'])
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

logsqrt2pi=np.log(np.sqrt(2.0 * np.pi))


for epoch in range(1,1+args.epoch):
    dataloader = DataLoader(buffer, batch_size=args.batch_size, shuffle=False)
    for idx, data in enumerate(dataloader):
        # data: obs,next_obs,action,reward,done
        obs,next_obs,action,reward,done=data
        with torch.no_grad():
            z,target_z=vae.encoder(obs),vae.encoder(next_obs)

        # logmix, mu, logstd, done_pred = model(z, data[2], data[4])# action,done
        logmix, mu, logstd, done_pred = model(z, action, done)# action,done
        # logmix = F.log_softmax(logmix)
        logmix_max = logmix.max(dim=1, keepdim=True)[0]
        logmix_reduce_logsumexp = (logmix - logmix_max).exp().sum(dim=1, keepdim=True).log() + logmix_max
        logmix = logmix - logmix_reduce_logsumexp

        # v = F.log_softmax(v)
        v = logmix - 0.5 * ((target_z - mu) / torch.exp(logstd)) ** 2 - logstd - logsqrt2pi
        v_max = v.max(dim=1, keepdim=True)[0]
        v = (v - v_max).exp().sum(dim=1).log() + v_max.squeeze()

        # maximize the prob, minimize the negative log likelihood
        z_loss = -v.mean()

        d_loss = F.binary_cross_entropy_with_logits(done_pred, done, reduce=False)
        r_factor = torch.ones_like(d_loss) # + done * args.rnn_rew_loss_weight # may no use for isaacgym
        d_loss = torch.mean(d_loss * r_factor)

        mse_loss = F.mse_loss(reward,rew_pred)
        loss = z_loss + d_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            info = "Epoch {:2d}\t Step [{:5d}/{:5d}]\t Z_Loss {:5.3f}\t \
                        R_Loss {:5.3f}\t Loss {:5.3f}\t ".format(
                epoch, idx, len(dataloader), z_loss.item(),
                d_loss.item(), loss.item())
            print(info)
    '''
    if epoch % 10 == 0:
        to_save_data = {'model': model.module.state_dict()}
        to_save_path = '{}/rnn_{}_e{:03d}.pth'.format(cfg.model_save_dir, cfg.timestr, epoch)
        torch.save(to_save_data, to_save_path)
    '''