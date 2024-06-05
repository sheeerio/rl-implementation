import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm

from vae import Encoder, Decoder, Model

dataset_path = './datasets/'
batch_size = 100
x_dim = 784
hidden_dim = 400
latent_dim = 200

lr = 1e-3
epochs = 30

mnist_transform = transforms.Compose([transforms.ToTensor()])
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = Model(Encoder=encoder, Decoder=decoder)

test_dataset = MNIST(dataset_path, transform = mnist_transform, train=False, download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model.load_state_dict(torch.load('vae_model.pth'))
model.eval()

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim)
        x_hat, _, _ = model(x)

        break

def show_image(x, idx):
    x = x.view(batch_size, 28, 28)
    fig = plt.figure()

    plt.imshow(x[idx].cpu().numpy())
    plt.savefig('output1.png')

show_image(x, idx=1)