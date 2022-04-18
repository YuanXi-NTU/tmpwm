""" Recurrent model training """
import argparse
import os
import easydict
import yaml
import torch

from functools import partial
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = easydict.EasyDict(yaml.load(open('../wm_config.yaml'), yaml.FullLoader))
# LSIZE below:  size of latent varialbe z
'''
parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")
# args.update(vars(parser.parse_args()))
'''

writer = SummaryWriter(args.log_path)


# params usage
vae_hidden_size = args.model.vae.hidden_size
latent_size = args.model.rnn.latent_size
rnn_size = args.model.rnn.rnn_size
num_mixtures = args.model.rnn.num_mixtures
seq_len = args.model.rnn.seq_len

lr = args.train.rnn.lr
batch_size = args.train.rnn.batch_size
epoch = args.train.rnn.epoch

# set model
vae = VAE(args.obs_shape, args.obs_shape, vae_hidden_size)

mdrnn = MDRNN(latent_size, args.action_shape, rnn_size, num_mixtures)
if args.loads.vae:
    vae.load_state_dict(torch.load(args.loads.vae_path)['vae'])
if args.loads.mdrnn:
    mdrnn.load_state_dict(torch.load(args.loads.mdrnn_path)['mdrnn'])
vae.to(device)
mdrnn.to(device)

optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=lr, alpha=.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=3)

path = args.data_path
train_loader = DataLoader(
    RolloutSequenceDataset(path, seq_len, train=True),
    batch_size=batch_size, num_workers=0, shuffle=True)
test_loader = DataLoader(
    RolloutSequenceDataset(path, seq_len, train=False),
    batch_size=batch_size, num_workers=0)


def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (batch_size, seq_len, action_shape, SIZE, SIZE)
    :args next_obs: 5D torch tensor (batch_size, seq_len, action_shape, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (batch_size, seq_len, LSIZE)
        - next_latent_obs: 4D torch tensor (batch_size, seq_len, LSIZE)
    """
    with torch.no_grad():
        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)]

        latent_obs, latent_next_obs = [
            # (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(batch_size, seq_len, latent_size)
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(-1, seq_len, latent_size)
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return latent_obs, latent_next_obs


def get_loss(latent_obs, action, reward, done,
             latent_next_obs, ):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (batch_size, seq_len, LSIZE) torch tensor
    :args action: (batch_size, seq_len, action_shape) torch tensor
    :args reward: (batch_size, seq_len) torch tensor
    :args latent_next_obs: (batch_size, seq_len, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    latent_obs, action, reward, done, latent_next_obs = \
        [arr.transpose(1, 0) for arr in
         [latent_obs, action, reward, done, latent_next_obs]]
    mus, sigmas, logpi, rew_pred, done_pred = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = F.binary_cross_entropy_with_logits(done_pred, done)
    mse = F.mse_loss(rew_pred, reward)
    # mse = torch.sum((rew_pred - reward)**2)

    # LSIZE/scale: for scale among different losses
    scale = vae_hidden_size + 2
    '''
    if include_reward:
        mse = f.mse_loss(rew_pred, reward)
        scale = LSIZE + 2
    else:
        mse = 0
        scale = LSIZE + 1
    '''

    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


def data_pass(epochs, train, logger_cnt):
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    print(epoch, logger_cnt)
    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0
    dataset_len = len(loader.dataset)
    pbar = tqdm(total=dataset_len, desc="Epoch {}".format(epochs))
    for i, data in enumerate(loader):
        obs, action, reward, done, next_obs = [arr.to(device) for arr in data]

        # transform obs
        latent_obs, latent_next_obs = to_latent(obs, next_obs)

        if train:
            losses = get_loss(latent_obs, action, reward,
                              done, latent_next_obs)

            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward,
                                  done, latent_next_obs)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item()
        if train:
            writer.add_scalar('train/loss', losses['loss'].item(), logger_cnt)
            writer.add_scalar('train/done_bce', losses['bce'].item(), logger_cnt)
            writer.add_scalar('train/gmm', losses['gmm'].item(), logger_cnt)
            writer.add_scalar('train/rew_mse', losses['mse'].item(), logger_cnt)
        else:
            writer.add_scalar('test/loss', losses['loss'].item(), logger_cnt)
            writer.add_scalar('test/done_bce', losses['bce'].item(), logger_cnt)
            writer.add_scalar('test/gmm', losses['gmm'].item(), logger_cnt)
            writer.add_scalar('test/rew_mse', losses['mse'].item(), logger_cnt)
        cum_loss=0
        logger_cnt += 1

        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} gmm={gmm:10.6f} mse={mse:10.6f}".format(
            loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
            gmm=cum_gmm / vae_hidden_size / (i + 1), mse=cum_mse / (i + 1)))
        pbar.update(batch_size)
    pbar.close()
    return cum_loss * batch_size / dataset_len, logger_cnt



if __name__=="__main__":
    cur_best = None
    logger_test_cnt = 0
    logger_train_cnt = 0
    for e in range(epoch):
        _, logger_train_cnt = data_pass(e, train=True, logger_cnt=logger_train_cnt)
        print('main',logger_train_cnt)
        test_loss, logger_test_cnt = data_pass(e, train=False, logger_cnt=logger_test_cnt)
        scheduler.step(test_loss)

        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss
            torch.save({'mdrnn': mdrnn.state_dict()}, 'mdrnn.pth')
