import torch
import torch.nn as nn

from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vq_vae import Encoder, Decoder, Model, VQEmbeddingEMA
import wandb

wandb.login()

dataset_path = "../datasets/"

mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# kwargs = {'num_workers': 1, 'pin_memory': True}

batch_size = 128
img_size = (32, 32)  # (width, height)

input_dim = 3
hidden_dim = 512
latent_dim = 16
n_embeddings = 512
output_dim = 3
commitment_beta = 0.25

lr = 2e-4

epochs = 50

print_step = 50

run = wandb.init(
    project="vq-vae",
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "commitment_beta": commitment_beta,
    },
)

train_dataset = CIFAR10(
    dataset_path, transform=mnist_transform, train=True, download=False
)
test_dataset = CIFAR10(
    dataset_path, transform=mnist_transform, train=False, download=False
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=latent_dim)
codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=latent_dim)
decoder = Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim)

model = Model(Encoder=encoder, Codebook=codebook, Decoder=decoder)

model.train()

mse_loss = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=lr)

print("Start training VQ-VAE")

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_hat, commitment_loss, codebook_loss, perplexity = model(x)
        recon_loss = mse_loss(x_hat, x)

        loss = recon_loss + commitment_loss + commitment_loss + codebook_loss

        loss.backward()
        optimizer.step()
        wandb.log(
            {
                "recon loss": recon_loss.item(),
                "perplexity": perplexity.item(),
                "commit loss": commitment_loss.item(),
                "codebook loss": codebook_loss.item(),
                "loss": loss.item(),
            }
        )
        if batch_idx % print_step == 0:
            print(
                "Epoch: ",
                epoch + 1,
                "(",
                batch_idx + 1,
                ") recon_loss: ",
                recon_loss.item(),
                "perplexity: ",
                perplexity,
                " commit_loss: ",
                commitment_loss.item(),
                "\n\t codebook loss: ",
                codebook_loss.item(),
                " total loss: ",
                loss.item(),
                "\n",
            )

torch.save(model.state_dict(), "vq-vae_model.pth")
print("Finish")
