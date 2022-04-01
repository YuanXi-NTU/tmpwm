
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
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
        raise ValueError(f"Unsupported initialization: {weight_init}")

class StateEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(StateEncoder, self).__init__()
        self.latent_size = latent_size
        self.input_size= input_size
        self.fc1 = nn.Linear(self.input_size, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.latent_size)
        self.fc_mu = nn.Linear(self.latent_size, self.latent_size)
        self.fc_logsigma  = nn.Linear(self.latent_size, self.latent_size)

    def forward(self, x): # pylint: disable=arguments-differ

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class StateDecoder(nn.Module):
    """ VAE decoder """
    def __init__(self, output_size, latent_size):
        super(StateDecoder, self).__init__()
        self.output_size= output_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(self.latent_size, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.output_size)


    def forward(self, x): 
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        reconstruction=F.sigmoid(x)
        return reconstruction


class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, input_size, output_size,latent_size):
        super(VAE, self).__init__()
        self.encoder = StateEncoder(input_size, latent_size)
        self.decoder = StateDecoder(output_size, latent_size)
        initialize_parameters(self.named_parameters(), 'orthogonal')
        self.bn=nn.BatchNorm1d(input_size)


    def forward(self, x): # pylint: disable=arguments-differ
        x=self.bn(x)
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
