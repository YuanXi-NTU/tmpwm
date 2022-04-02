import easydict,yaml
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader
from models import MDRNN, gmm_loss, VAE

args=easydict.EasyDict(yaml.load(open('./rnn_config.yaml'),yaml.FullLoader))

model = MDRNN(args.vae_latent_size, args.action_shape, args.latent_size, args.num_mixtures).cuda()
vae= VAE(args.vae_latent_size).cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

logsqrt2pi=np.log(np.sqrt(2.0 * np.pi))
for epoch in range(1,1+args.epoch):
    dataloader = DataLoader(buffer, batch_size=args.batch_size, shuffle=False)
    for idx, data in enumerate(dataloader):
        # reward, done, action,obs, next_obs
        obs,next_obs,action,reward,done=data
        with torch.no_grad():
            z,target_z=vae.encoder(obs),vae.encoder(next_obs)

        logmix, mu, logstd, done_pred = model(z, data[2], data[4])
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

        r_loss = F.binary_cross_entropy_with_logits(done_pred, done, reduce=False)
        r_factor = torch.ones_like(r_loss) + done * args.rnn_rew_loss_weight
        r_loss = torch.mean(r_loss * r_factor)

        loss = z_loss + r_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            info = "Epoch {:2d}\t Step [{:5d}/{:5d}]\t Z_Loss {:5.3f}\t \
                        R_Loss {:5.3f}\t Loss {:5.3f}\t ".format(
                epoch, idx, len(dataloader), z_loss.item(),
                r_loss.item(), loss.item())
            print(info)
    '''
    if epoch % 10 == 0:
        to_save_data = {'model': model.module.state_dict()}
        to_save_path = '{}/rnn_{}_e{:03d}.pth'.format(cfg.model_save_dir, cfg.timestr, epoch)
        torch.save(to_save_data, to_save_path)
    '''