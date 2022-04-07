import os
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
import easydict,yaml

from models.vae import VAE
from utils.learning import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorboardX import SummaryWriter
writer=SummaryWriter('./log/')

args=easydict.EasyDict(yaml.load(open('./wm_config.yaml'),yaml.FullLoader))

'''
import argparse
parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                    help='input batch size for training (default: 32)')
'''


cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

# params usage
hidden_size=args.model.vae.hidden_size
batch_size=args.train.vae.batch_size
lr=args.train.vae.lr
epoch=args.train.vae.epoch
model = VAE(args.obs_shape, args.obs_shape,
            hidden_size).to(device)
path='/data/yuanxi20/res_buffer.pickle'
dataset_train = RolloutObservationDataset(path, train=True)
dataset_test = RolloutObservationDataset(path, train=False)


train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)



optimizer = optim.Adam(model.parameters(),lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=2)
earlystopping = EarlyStopping('min', patience=10)

global batch_num_train
global batch_num_test

batch_num_train=0
batch_num_test=0

def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    # BCE = F.mse_loss(recon_x, x)
    BCE=torch.mean((recon_x-x)**2)*100
    # KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    KLD = -0.5 * torch.mean(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())*100
    return BCE + KLD,BCE,KLD


def train(epochs,logger_cnt):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):

        # obs,action,next_obs=data[0],data[1],data[2]
        # input=torch.cat([obs,action],dim=1)
        # input,next_obs=input.to(device),next_obs.to(device)
        # obs,next_obs=obs.to(device),next_obs.to(device)

        obs=data[0]        
        obs=obs.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(obs)
        # loss,bce,kld = loss_function(recon_batch, next_obs, mu, logvar)
        
        loss,bce,kld = loss_function(recon_batch, obs, mu, logvar)
        writer.add_scalar('losses/train_loss', loss.item(), logger_cnt)
        writer.add_scalar('losses/train_bce', bce.item(), logger_cnt)
        writer.add_scalar('losses/train_kld', kld.item(), logger_cnt)
        logger_cnt += 1
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f},bce: {:.6f},kld: {:.6f}'.format(
                epochs,100. * batch_idx / len(train_loader),
                loss.item() / len(obs),bce.item()/len(obs),kld.item()/len(obs)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epochs, train_loss / len(train_loader.dataset)))
    return logger_cnt


def test(logger_cnt):
    model.eval()
    test_loss,test_kld,test_bce = [],[],[]
    with torch.no_grad():
        for data in test_loader:

            obs, action, next_obs = data[0], data[1], data[2]

            # input = torch.cat([obs, action], dim=1)
            # input,next_obs=input.to(device),next_obs.to(device)
            obs,next_obs=obs.to(device),next_obs.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(obs)
            loss,bce,kld = loss_function(recon_batch, next_obs, mu, logvar)

            writer.add_scalar('losses/test_loss', loss.item(), logger_cnt)
            writer.add_scalar('losses/test_bce', bce.item(), logger_cnt)
            writer.add_scalar('losses/test_kld', kld.item(), logger_cnt)
            logger_cnt += 1

            test_loss.append(loss.item())
            test_kld.append(kld.item())
            test_bce.append(bce.item())

    test_loss =torch.mean(torch.tensor(test_loss))
    test_kld =torch.mean(torch.tensor(test_kld))
    test_bce =torch.mean(torch.tensor(test_bce))
    print('====> Test set loss: {:.4f},bce&kld: {:.4f},{:.4f}'.format(test_loss,bce,kld))
    return test_loss,logger_cnt


cur_best = None
logger_test_cnt = 0
logger_train_cnt = 0
for epoch in range(1, epoch + 1):

    logger_train_cnt=train(epoch,logger_train_cnt)
    test_loss,logger_test_cnt= test(logger_test_cnt)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
        torch.save({'vae':model.state_dict()},'vae_.pth')

    '''#used by previous author
    
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

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
    '''

