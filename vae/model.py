import torch
from torch import nn
from torch.nn import functional as F

class AutoEncoder(nn.Module):
    def __init__(self, config):
        pass

    def encode(self, ):
        pass
    
    def decode(self, ):
        pass

    def forward(self, ):
        pass

class LstmAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        config.input_dim
        config.hidden_dim
        config.latent_dim

        self.bidirectional = config.bidirectional
        self.layer = config.layer

        #encoder struc
        self.encoder_lstm = nn.LSTM(config.input_dim, config.hidden_dim, bidirectional=config.bidirectional, batch_first=True)
        self.encoder_linear = nn.Linear(config.hidden_dim, config.latent_dim)

        #decoder struc
        self.decoder_lstm = nn.LSTM(config.latent_dim, config.latent_dim, bidirectional=config.bidirectional, batch_first=True)

    def encode(self, ):
        pass

    def decode(self, ):
        pass

    def forward(self, xs): # batch_size * seq_length * dim
        batch_size = x.shape[0]

        return_xs = []
        for i, x in enumerate(xs):
            z = self.encode(x)
            return_x = self.decode(z)
            return_xs.append(return_x)
        return z, return_xs

    @classmethod
    def loss_function(cls, xs, return_xs, z):
        pass


class LstmVariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.vocab_size = config.vocab_size

        self.bidirectional = config.bidirectional
        self.layer = config.layer

        #encoder struc
        self.encoder_lstm = nn.LSTM(config.input_dim, config.hidden_dim, bidirectional=config.bidirectional, num_layers=config.layer, batch_first=True)
        if config.bidirectional is True:
            self.encoder_fc_mu = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.latent_dim),
                nn.ReLU(inplace=True)
            )
            self.encoder_fc_logvar = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.latent_dim),
                nn.ReLU(inplace=True)
            )
            # self.encoder_fc_mu = nn.Linear(config.hidden_dim * 2, config.latent_dim)
            # self.encoder_fc_logvar = nn.Linear(config.hidden_dim * 2, config.latent_dim)
        else:
            self.encoder_fc_mu = nn.Sequential(
                nn.Linear(config.hidden_dim, config.latent_dim),
                nn.ReLU(inplace=True)
            )
            self.encoder_fc_logvar = nn.Sequential(
                nn.Linear(config.hidden_dim, config.latent_dim),
                nn.ReLU(inplace=True)
            )
            # self.encoder_fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
            # self.encoder_fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        #decoder struc
        self.decoder_lstm = nn.LSTM(config.input_dim, config.latent_dim, bidirectional=config.bidirectional, num_layers=config.layer, batch_first=True)
        if config.bidirectional is True:
            # self.decoder_linear = nn.Sequential(
            #     nn.Linear(config.latent_dim * 2, config.vocab_size),
            #     nn.ReLU(inplace=True)
            # )
            self.decoder_linear = nn.Linear(config.latent_dim * 2, config.vocab_size)
        else:
            # self.decoder_linear = nn.Sequential(
            #     nn.Linear(config.latent_dim, config.vocab_size),
            #     nn.ReLU(inplace=True)
            # )        
            self.decoder_linear = nn.Linear(config.latent_dim, config.vocab_size)            

    def encode(self, x):
        output, (hidden_states, cell_states) = self.encoder_lstm(x) # hidden_states: batch_size * layer * hd
        if self.bidirectional is True:
            hidden_states = torch.cat(hidden_states.split(self.layer, dim = 1), dim = 2)
        mu = self.encoder_fc_mu(hidden_states)
        logvar = self.encoder_fc_logvar(hidden_states)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def decode(self, z, length, inputs=None):
        batch_size = z.shape[1]

        outputs = []
        init_input = torch.zeros(batch_size, self.input_dim)
        hidden_states = z
        cell_states = torch.zeros(self.layer, batch_size, self.latent_dim)

        if inputs is not None:
            predict_inputs = torch.cat([init_input, inputs[:-1]]).unsqueeze(0)
            outputs, (hidden_states, cell_states) = self.decoder_lstm(predict_inputs, (hidden_states, cell_states))
            outputs = torch.softmax(self.decoder_linear(outputs).squeeze(0), dim = 1)
        else:
            pass
    
        return outputs

    def forward(self, inputs, **kwargs):
        mus = []
        logvars = []
        for seq in inputs:
            mu, logvar = self.encode(seq.unsqueeze(0))
            mus.append(mu.unsqueeze(0))
            logvars.append(logvar.unsqueeze(0))
        mus = torch.cat(mus)
        logvars = torch.cat(logvars)
        z = self.reparameterize(mus, logvars)

        recons = []
        for i in range(z.shape[0]):
            recon = self.decode(z[i], len(inputs[i]), inputs[i])
            recons.append(recon)
        return recons, z, mus, logvars

    def loss_function(self, inputs, lens, recons, mus, logvars, kld_weight):
        batch_size = len(inputs)
        
        logvars = logvars.view(batch_size, -1)
        mus = mus.view(batch_size, -1)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvars - mus ** 2 - logvars.exp(), dim = 1), dim = 0)

        recons_loss = 0
        for i, (recon, input) in enumerate(zip(recons, inputs)):
            recons_loss += F.cross_entropy(recon, input[:lens[i]])
        recons_loss /= len(inputs)

        loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'recons_loss': recons_loss,
            'kld_loss': kld_loss
        }
