import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.input1 = nn.Linear(input_dim, hidden_dim)
        self.input2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.training = True

    def forward(self, x):
        x = self.leakyrelu(self.input1(x))
        x = self.leakyrelu(self.input2(x))
        mean = self.mean(x)
        log_var = self.var(x)

        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        self.hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leakyrelu(self.hidden(x))
        x = self.leakyrelu(self.hidden2(x))
        x_hat  = torch.sigmoid(self.output(x))
        return x_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + epsilon * var
        return z
    
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)
        return x_hat, mean, log_var