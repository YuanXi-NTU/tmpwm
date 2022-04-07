import torch
import torch.nn as nn
import torch.nn.functional as F
def initialize_parameters(params, mode):
    if mode == "default":
        pass
    elif mode == "orthogonal":
        for name, param in params:
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                param.data.fill_(0)
    elif mode == "normal":
        for name, param in params:
            if len(param.shape) >= 2:
                torch.nn.init.normal_(param)
            elif "bias" in name:
                param.data.fill_(0)
    elif mode == "uniform":
        for name, param in params:
            if len(param.shape) >= 2:
                torch.nn.init.uniform_(param)
            elif "bias" in name:
                param.data.fill_(0)
    else:
        raise ValueError(f"Unsupported initialization")

class StateEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(StateEncoder, self).__init__()
        self.latent_size = latent_size
        self.input_size= input_size
        self.bn1=nn.BatchNorm1d(input_size)
        self.bn2=nn.BatchNorm1d(latent_size)

        self.fc1 = nn.Linear(self.input_size, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, self.latent_size)
        self.fc_mu = nn.Linear(self.latent_size, self.latent_size)
        self.fc_logsigma  = nn.Linear(self.latent_size, self.latent_size)

    def forward(self, x):
        cur_shape=x.shape
        # x=self.bn1(x.view(-1,self.input_size))
        x=self.bn1(x)
        x=F.relu(self.fc1(x))
        x=self.bn2(x)
        x=self.fc2(x)
        mu = F.sigmoid(self.fc_mu(x))
        # mu = F.sigmoid(self.fc_mu(x)).view(cur_shape[0],self.latent_size)
        logsigma = F.sigmoid(self.fc_logsigma(x)).view(cur_shape[0],self.latent_size)

        return mu, logsigma

class StateDecoder(nn.Module):
    def __init__(self, output_size, latent_size):
        super(StateDecoder, self).__init__()
        self.output_size= output_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(self.latent_size, self.latent_size)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(latent_size)
        # self.fc2 = nn.Linear(self.latent_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.output_size)


    def forward(self, x):
        cur_shape=x.shape
        x=F.relu(self.fc1(x))
        # x=F.relu(self.fc1(x.view(-1,self.latent_size)))
        x=self.bn2(x)
        # x=F.relu(self.fc2(x))
        x=self.fc3(x)
        # x=self.bn1(x).view(cur_shape[0],cur_shape[1],self.output_size)
        x=self.bn1(x)
        # x=self.bn1(x).view(cur_shape[0],self.output_size)
        reconstruction=F.sigmoid(x)
        return reconstruction


class VAE(nn.Module):
    def __init__(self, input_size, output_size,latent_size):
        super(VAE, self).__init__()
        self.encoder = StateEncoder(input_size, latent_size)
        self.decoder = StateDecoder(output_size, latent_size)
        initialize_parameters(self.named_parameters(), 'orthogonal')


    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
