import argparse,os
from functools import partial
import easydict,yaml
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader

from tqdm import tqdm
# from utils.misc import save_checkpoint
# from utils.learning import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ASIZE, LSIZE, RSIZE = 3, 32, 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args=easydict.EasyDict(yaml.load(open('./rnn_config.yaml'),yaml.FullLoader))
'''
parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")
# args = parser.parse_args()
args.update(vars(parser.parse_args()))
'''

# set model
vae= VAE(args.obs_shape,args.obs_shape,args.model.vae_latent_size).to(device)
vae_path='./vae.pth'
vae.load_state_dict(torch.load(vae_path)['vae'])
mdrnn = MDRNN(args.model.latent_size, args.action_shape, RSIZE, 5)#RSIZE: rnn size
mdrnn.to(device)

optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=args.lr, alpha=.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
# earlystopping = EarlyStopping('min', patience=30)

path='/home/yuanxi20/isaacgym/IsaacGymEnvs/isaacgymenvs/buffer_data/res_buffer.pickle'

train_loader = DataLoader(
    RolloutSequenceDataset(path, args.model.seq_len, train=True),
    batch_size=args.batch_size, num_workers=4, shuffle=True)
test_loader = DataLoader(
    RolloutSequenceDataset(path, args.model.seq_len, train=False),
    batch_size=args.batch_size, num_workers=4)

def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (args.batch_size, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (args.batch_size, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (args.batch_size, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (args.batch_size, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)]

        latent_obs, latent_next_obs = [
            # (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(args.batch_size, args.model.seq_len, args.model.latent_size)
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(-1, args.model.seq_len, args.model.latent_size)
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return latent_obs, latent_next_obs

def get_loss(latent_obs, action, reward, done,
             latent_next_obs,):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (args.batch_size, SEQ_LEN, LSIZE) torch tensor
    :args action: (args.batch_size, SEQ_LEN, ASIZE) torch tensor
    :args reward: (args.batch_size, SEQ_LEN) torch tensor
    :args latent_next_obs: (args.batch_size, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    latent_obs, action, reward, done, latent_next_obs = \
        [arr.transpose(1, 0) for arr in
         [latent_obs, action, reward, done, latent_next_obs]]
    mus, sigmas, logpi, rew_pred, done_pred = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = f.binary_cross_entropy_with_logits(done_pred, done)
    mse = f.mse_loss(rew_pred, reward)
    scale = LSIZE + 2
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


def data_pass(epoch, train):
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader


    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
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

        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                             "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                                 loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                 gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1)))
        pbar.update(args.batch_size)
    pbar.close()
    return cum_loss * args.batch_size / len(loader.dataset)


train = partial(data_pass, train=True)
test = partial(data_pass, train=False)

cur_best = None
for e in range(args.epoch):
    train(e)
    test_loss = test(e)
    scheduler.step(test_loss)
    # earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
        torch.save({'mdrnn':mdrnn.state_dict()},'mdrnn.pth')
    '''
    checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                    rnn_file)
    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(e))
        break                    
'''

