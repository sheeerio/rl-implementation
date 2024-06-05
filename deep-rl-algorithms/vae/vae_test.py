import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vae import Encoder, Decoder, Model

dataset_path = './datasets/'
batch_size = 100
x_dim = 784
hidden_dim = 400
latent_dim = 200

lr = 1e-3
epochs = 30

mnist_transform = transforms.Compose([transforms.ToTensor()])

# kwargs = {'num_workers': 1, 'pin_memory': True}

train_dataset = MNIST(dataset_path, transform = mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform = mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

model = Model(Encoder=encoder, Decoder=decoder)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kl_div

optimizer = Adam(model.parameters(), lr=lr)

print("Start training VAE...")
model.train()

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    print("\tEpoch ", epoch + 1, "complete", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
print("Finish")