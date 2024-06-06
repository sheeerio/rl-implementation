import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

dataset_path = "./datasets/"

batch_size = 128
img_size = (32, 32)

input_dim = 128
img_size = (32, 32)

input_dim = 3
hidden_dim = 512
latent_dim = 16
n_embeddings = 512
output_dim = 3
commitment_beta = 0.25

lr = 2e-4

epochs = 50
print_step = 50

mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# kwargs = {'num_workers': 1, 'pin_memory': True}

train_dataset = CIFAR10(
    dataset_path, transform=mnist_transform, train=True, download=True
)
test_dataset = CIFAR10(
    dataset_path, transform=mnist_transform, train=False, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Encoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, kernel_size=(4, 4, 3, 1), stride=2
    ):
        super(Encoder, self).__init__()
        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_size
        self.strided_conv_1 = nn.Conv2d(
            input_dim, hidden_dim, kernel_1, stride, padding=1
        )
        self.strided_conv_2 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_2, stride, padding=1
        )

        self.residual_conv_1 = nn.Cond2d(hidden_dim, hidden_dim, kernel_3, padding=1)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_4, padding=0)

        self.proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = self.strided_conv_1(x)
        x = self.strided_conv_2(x)

        x = F.relu(x)
        y = self.residual_conv_1(x)
        y = y + x

        x = F.relu(y)
        y = self.residual_conv_2(x)
        y = y + x

        return y


class VQEmbeddingEMA(nn.Module):
    def __init__(
        self,
        n_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        _, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view_as(x)

    def retrieve_random_codebook(self, random_indices):
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)
        return quantized

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2
        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, num_classes=M).float()
        quantized = F.embedding(indices, self.embedding).view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(
                encodings, dim=0
            )
            n = torch.sum(self.ema_count)
