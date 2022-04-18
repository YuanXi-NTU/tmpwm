import os
import torch
from torch.nn import functional as F
import easydict
import yaml
import pickle
from tensorboardX import SummaryWriter

from models.vae import VAE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

args = easydict.EasyDict(yaml.load(open('../wm_config.yaml'), yaml.FullLoader))

'''
import argparse
parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                    help='input batch size for training (default: 32)')
# args.update(vars(parser.parse_args()))

'''

writer = SummaryWriter(args.log_path)

# cuda settings 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if cuda else "cpu")
torch.set_printoptions(precision=5, sci_mode=False)

# params usage
hidden_size = args.model.vae.hidden_size
batch_size = args.train.vae.batch_size
lr = args.train.vae.lr
epoch = args.train.vae.epoch
model = VAE(args.obs_shape, args.obs_shape,
            hidden_size).to(device)

dataset_train = RolloutObservationDataset(args.data_path, train=True)
dataset_test = RolloutObservationDataset(args.data_path, train=False)

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

if args.loads.vae:
    model.load_state_dict(torch.load(args.loads.vae_path))


def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    MSE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())

    # return MSE/MSE.item() + KLD / KLD.item(), MSE, KLD
    return MSE, MSE, KLD


def train(epochs, logger_cnt):
    model.train()
    train_loss = []
    for batch_idx, data in enumerate(train_loader):
        obs = data
        obs = obs.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(obs)

        loss, mse, kld = loss_function(recon_batch, obs, mu, logvar)
        loss.backward()

        writer.add_scalar('losses/train_loss', loss.item(), logger_cnt)
        writer.add_scalar('losses/train_mse', mse.item(), logger_cnt)
        writer.add_scalar('losses/train_kld', kld.item(), logger_cnt)
        logger_cnt += 1

        train_loss.append(loss.item())
        u = torch.tensor(model.encoder.fc1.weight)

        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}\tmse: {:.6f} \tkld: {:.6f}'.format(
                epochs, 100. * batch_idx / len(train_loader),
                loss.item(), mse.item(), kld.item()))
        # if batch_idx % 160 == 0:
        #         print(obs[0,:10])
        #         print(recon_batch[0,:10].detach())
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epochs, torch.mean(torch.tensor(train_loss)).item()))
    return logger_cnt


def test(logger_cnt):
    model.eval()
    test_loss, test_kld, test_mse = [], [], []
    with torch.no_grad():
        for data in test_loader:
            obs = data[0]
            obs = obs.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(obs)
            loss, mse, kld = loss_function(recon_batch, obs, mu, logvar)

            writer.add_scalar('losses/test_loss', loss.item(), logger_cnt)
            writer.add_scalar('losses/test_mse', mse.item(), logger_cnt)
            writer.add_scalar('losses/test_kld', kld.item(), logger_cnt)
            logger_cnt += 1

            test_loss.append(loss.item())
            test_kld.append(kld.item())
            test_mse.append(mse.item())

    test_loss = torch.mean(torch.tensor(test_loss))
    test_kld = torch.mean(torch.tensor(test_kld))
    test_mse = torch.mean(torch.tensor(test_mse))
    print('====> Test set loss: {:.4f}, mse:{:.4f}, kld:{:.4f}'.format(test_loss, test_mse, test_kld))
    return test_loss, logger_cnt


if __name__ == "__main__":
    cur_best = None
    logger_test_cnt = 0
    logger_train_cnt = 0
    for e in range(1, epoch + 1):

        logger_train_cnt = train(e, logger_train_cnt)
        test_loss, logger_test_cnt = test(logger_test_cnt)
        # scheduler.step(test_loss)

        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss
            torch.save({'vae': model.state_dict()}, 'vae.pth')
