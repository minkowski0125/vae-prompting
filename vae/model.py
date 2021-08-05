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

class TransferLayer(nn.Module):
    def __init__(self, input_dim, output_dim, layer_num, config):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.device = config.device

        self.layers = []
        self.add_layer(input_dim, output_dim)
        for i in range(layer_num - 1):
            self.add_layer(output_dim, output_dim)

class LinearTransferLayer(TransferLayer):
    def __init__(self, input_dim, output_dim, layer_num, config):
        super(LinearTransferLayer, self).__init__(input_dim, output_dim, layer_num, config)

    def add_layer(self, input_dim, output_dim):
        self.layers.append(
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(inplace=True)
            ).to(self.device)
        )
    
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

class HighwayTransferLayer(TransferLayer):
    def __init__(self, input_dim, output_dim, layer_num, config):
        super(HighwayTransferLayer, self).__init__(input_dim, output_dim, layer_num, config)

    def add_layer(self, input_dim, output_dim):
        H = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True)
        ).to(self.device)

        T = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        ).to(self.device)
        self.layers.append((H,T))
        
    
    def forward(self, x):
        output = x
        for H, T in self.layers:
            h, t = H(output), T(output)
            output = torch.mul(h, t) + torch.mul(h, (1 - t))
        return output

class TranslationAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self, x):
        return None

class LstmAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.device = config.device
        self.reconstruction = config.reconstruction

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.vocab_size = config.vocab_size

        self.bidirectional = config.bidirectional
        self.layer = config.layer

        #encoder struc
        self.encoder_lstm = nn.LSTM(config.input_dim, config.hidden_dim, bidirectional=config.bidirectional, num_layers=config.layer, batch_first=True)
        if config.bidirectional is True:
            if config.transfer == 'linear':
                self.encoder_transfer_layer = LinearTransferLayer(config.hidden_dim * 2, config.latent_dim, 4, config)
                self.decoder_transfer_layer = LinearTransferLayer(config.latent_dim, config.hidden_dim, 4, config)
            elif config.transfer == 'highway':
                self.encoder_transfer_layer = HighwayTransferLayer(config.hidden_dim * 2, config.latent_dim, 4, config)
                self.decoder_transfer_layer = HighwayTransferLayer(config.latent_dim, config.hidden_dim, 4, config)
        else:
            if config.transfer == 'linear':
                self.encoder_transfer_layer = LinearTransferLayer(config.hidden_dim, config.latent_dim, 4, config)
                self.decoder_transfer_layer = LinearTransferLayer(config.latent_dim, config.hidden_dim, 4, config)
            elif config.transfer == 'highway':
                self.encoder_transfer_layer = HighwayTransferLayer(config.hidden_dim, config.latent_dim, 4, config)
                self.decoder_transfer_layer = HighwayTransferLayer(config.latent_dim, config.hidden_dim, 4, config)

        #decoder struc
        self.decoder_lstm = nn.LSTM(config.input_dim, config.hidden_dim, bidirectional=config.bidirectional, num_layers=config.layer, batch_first=True)
   
        if config.reconstruction == 'discrete':
            self.decoder_linear = nn.Linear(config.latent_dim, config.vocab_size)            
        elif config.reconstruction == 'continuous':
            self.decoder_linear = nn.Linear(config.latent_dim, config.input_dim)

    def encode(self, x):
        output, (hidden_states, cell_states) = self.encoder_lstm(x) # hidden_states: batch_size * layer * hd
        if self.bidirectional is True:
            hidden_states = torch.cat(hidden_states.split(self.layer, dim = 1), dim = 2)
        z = self.encoder_transfer_layer(hidden_states).to(self.device)

        return z

    def decode(self, z, length, inputs=None):
        batch_size = z.shape[1]

        outputs = []
        # init_input = torch.zeros(batch_size, self.input_dim).to(self.device)
        # print(inputs.shape)
        # raw_inputs = torch.zeros(1, length, self.input_dim).to(self.device)
        hidden_states = self.decoder_transfer_layer(z)
        cell_states = torch.zeros(self.layer, batch_size, self.latent_dim).to(self.device)

        if inputs is not None:
            predict_inputs = inputs[:-1].unsqueeze(0)
            # predict_inputs = raw_inputs
            outputs, (hidden_states, cell_states) = self.decoder_lstm(predict_inputs, (hidden_states, cell_states))
            outputs = self.decoder_linear(outputs).squeeze(0).to(self.device)
        else:
            pass
    
        return outputs

    def forward(self, inputs, **kwargs):
        zs = []
        for seq in inputs:
            z = self.encode(seq.unsqueeze(0))
            zs.append(z.unsqueeze(0))
        zs = torch.cat(zs)

        recons = []
        for i in range(zs.shape[0]):
            recon = self.decode(zs[i], len(inputs[i]), inputs[i])
            recons.append(recon)
        return recons, zs

    def loss_function(self, input_ids, input_seqs, lens, recons):
        batch_size = len(input_ids)
        
        if self.reconstruction == 'discrete':
            recons_loss = 0
            for i, (recon, input) in enumerate(zip(recons, input_ids)):
                recons_loss += F.cross_entropy(recon, input[1:lens[i]])
            recons_loss /= batch_size
        elif self.reconstruction == 'continuous':
            recons_loss = 0
            for i, (recon, input) in enumerate(zip(recons, input_seqs)):
                recons_loss += F.mse_loss(recon, input[1:])
            recons_loss /= batch_size

        loss = recons_loss
        return {
            'loss': loss,
            'recons_loss': recons_loss,
        }

class LstmVariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.device = config.device
        self.reconstruction = config.reconstruction

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.vocab_size = config.vocab_size

        self.bidirectional = config.bidirectional
        self.layer = config.layer

        #encoder struc
        self.encoder_lstm = nn.LSTM(config.input_dim, config.hidden_dim, bidirectional=config.bidirectional, num_layers=config.layer, batch_first=True)
        if config.bidirectional is True:
            if config.transfer == 'linear':
                self.encoder_transfer_mu = LinearTransferLayer(config.hidden_dim * 2, config.latent_dim, 4, config)
                self.encoder_transfer_logvar = LinearTransferLayer(config.hidden_dim * 2, config.latent_dim, 4, config)
                self.decoder_transfer_layer = LinearTransferLayer(config.latent_dim, config.hidden_dim, 4, config)
            elif config.transfer == 'highway':
                self.encoder_transfer_mu = HighwayTransferLayer(config.hidden_dim * 2, config.latent_dim, 4, config)
                self.encoder_transfer_logvar = HighwayTransferLayer(config.hidden_dim * 2, config.latent_dim, 4, config)
                self.decoder_transfer_layer = HighwayTransferLayer(config.latent_dim, config.hidden_dim, 4, config)
        else:
            if config.transfer == 'linear':
                self.encoder_transfer_mu = LinearTransferLayer(config.hidden_dim, config.latent_dim, 4, config)
                self.encoder_transfer_logvar = LinearTransferLayer(config.hidden_dim, config.latent_dim, 4, config)
                self.decoder_transfer_layer = LinearTransferLayer(config.latent_dim, config.hidden_dim, 4, config)
            elif config.transfer == 'highway':
                self.encoder_transfer_mu = HighwayTransferLayer(config.hidden_dim, config.latent_dim, 4, config)
                self.encoder_transfer_logvar = HighwayTransferLayer(config.hidden_dim, config.latent_dim, 4, config)
                self.decoder_transfer_layer = HighwayTransferLayer(config.latent_dim, config.hidden_dim, 4, config)

        #decoder struc
        self.decoder_lstm = nn.LSTM(config.input_dim, config.latent_dim, bidirectional=config.bidirectional, num_layers=config.layer, batch_first=True)     
        if config.reconstruction == 'discrete':
            self.decoder_linear = nn.Linear(config.latent_dim, config.vocab_size)            
        elif config.reconstruction == 'continuous':
            self.decoder_linear = nn.Linear(config.latent_dim, config.input_dim)            

    def encode(self, x):
        output, (hidden_states, cell_states) = self.encoder_lstm(x) # hidden_states: batch_size * layer * hd
        if self.bidirectional is True:
            hidden_states = torch.cat(hidden_states.split(self.layer, dim = 1), dim = 2)
        mu = self.encoder_transfer_mu(hidden_states).to(self.device)
        logvar = self.encoder_transfer_logvar(hidden_states).to(self.device)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def decode(self, z, length, inputs=None):
        batch_size = z.shape[1]

        outputs = []
        # init_input = torch.zeros(batch_size, self.input_dim).to(self.device)
        # print(inputs.shape)
        raw_inputs = torch.zeros(1, length, self.input_dim).to(self.device)
        hidden_states = self.decoder_transfer_layer(z)
        cell_states = torch.zeros(self.layer, batch_size, self.latent_dim).to(self.device)

        if inputs is not None:
            predict_inputs = inputs[:-1].unsqueeze(0)
            # predict_inputs = raw_inputs
            outputs, (hidden_states, cell_states) = self.decoder_lstm(predict_inputs, (hidden_states, cell_states))
            outputs = self.decoder_linear(outputs).squeeze(0).to(self.device)
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

    def loss_function(self, input_ids, input_seqs, lens, recons, mus, logvars, kld_weight):
        batch_size = len(input_ids)
        
        logvars = logvars.view(batch_size, -1)
        mus = mus.view(batch_size, -1)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvars - mus ** 2 - logvars.exp(), dim = 1), dim = 0)

        if self.reconstruction == 'discrete':
            recons_loss = 0
            for i, (recon, input) in enumerate(zip(recons, input_ids)):
                recons_loss += F.cross_entropy(recon, input[1:lens[i]])
            recons_loss /= batch_size
        elif self.reconstruction == 'continuous':
            recons_loss = 0
            for i, (recon, input) in enumerate(zip(recons, input_seqs[1:])):
                recons_loss += F.mse_loss(recon, input)
            recons_loss /= batch_size

        loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'recons_loss': recons_loss,
            'kld_loss': kld_loss
        }
