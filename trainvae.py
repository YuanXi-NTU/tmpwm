import os
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
import easydict,yaml

# from utils.misc import LSIZE, RED_SIZE
from models.vae import VAE
from utils.learning import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
import argparse
parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--hidden-size', type=int, default=64, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
args = parser.parse_args()
'''

args=easydict.EasyDict(yaml.load(open('./vae_config.yaml'),yaml.FullLoader))
cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

model = VAE(args.obs_shape, args.obs_shape,
            args.model.hidden_size).to(device)

path='/home/yuanxi20/isaacgym/IsaacGymEnvs/isaacgymenvs/buffer_data/res_buffer.pickle'
dataset_train = RolloutObservationDataset(path,None, train=True)
dataset_test = RolloutObservationDataset(path,None, train=False)


train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)



optimizer = optim.Adam(model.parameters(),lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD,BCE,KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):

        obs,action,next_obs=data[0],data[1],data[2]
        
        input=torch.cat([obs,action],dim=1)
        input,next_obs=input.to(device),next_obs.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(input)
        loss,bce,kld = loss_function(recon_batch, next_obs, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},{:.6f},{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(input),bce.item()/len(input),kld.item()/len(input)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:

            obs, action, next_obs = data[0], data[1], data[2]
            input = torch.cat([obs, action], dim=1)
            input,next_obs=input.to(device),next_obs.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(input)
            loss,bce,kld = loss_function(recon_batch, next_obs, mu, logvar)
            test_loss+=loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f},bce&kld: {:.4f},{:.4f}'.format(test_loss,bce,kld))
    return test_loss


cur_best = None

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
        torch.save({'vae':model.stat_dict(),'vae.pth'})

    '''
    # save_checkpoint({
    #     'epoch': epoch,
    #     'state_dict': model.state_dict(),
    #     'precision': test_loss,
    #     'optimizer': optimizer.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     'earlystopping': earlystopping.state_dict()
    # }, is_best, filename, best_filename)

    if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(RED_SIZE, LSIZE).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 3, RED_SIZE, RED_SIZE),
                       join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))
    '''

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
