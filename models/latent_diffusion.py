import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_var = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, out_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))  # Output in range [-1, 1]
        return x


class VAE(pl.LightningModule):
    def __init__(self, latent_dim=16, kl_weight=0.001, lr=1e-3):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.lr = lr

        self.encoder = Encoder(in_channels=1, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

    def encode(self, x):
        mu, log_var = self.encoder(x)
        return self.reparameterize(mu, log_var)

    def decode(self, z):
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstruction, mu, log_var = self(x)

        recon_loss = F.mse_loss(reconstruction, x)

        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / (x.size(0) * x.size(1) * x.size(2) * x.size(3))

        loss = recon_loss + self.kl_weight * kl_loss

        self.log("train_loss", loss)
        self.log("recon_loss", recon_loss)
        self.log("kl_loss", kl_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) float tensor of timesteps.
        device = t.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb  # (B, dim)


class LatentBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        # Use a projection if input/output dimensions differ
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = F.relu(self.fc1(x))
        # Add time embedding
        h = h + self.time_mlp(t_emb)
        h = F.relu(self.fc2(h))
        return h + self.shortcut(x)


# -----------------------------
# Latent Diffusion U-Net (Simpler Architecture for Latent Space)
# -----------------------------
class LatentUNet(nn.Module):
    def __init__(self, latent_dim=16, time_emb_dim=128, hidden_dim=128):
        super().__init__()
        # Time embedding
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.net = nn.ModuleList([
            LatentBlock(latent_dim, hidden_dim, hidden_dim),
            LatentBlock(hidden_dim, hidden_dim, hidden_dim),
            LatentBlock(hidden_dim, hidden_dim, hidden_dim),
            LatentBlock(hidden_dim, latent_dim, hidden_dim)
        ])

    def forward(self, x, t):
        # Compute time embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Forward through network
        h = x
        for block in self.net:
            h = block(h, t_emb)
        return h


class LatentDiffusion(pl.LightningModule):
    def __init__(self, vae, timesteps=1000, beta_start=1e-4, beta_end=0.02, lr=1e-4):
        super().__init__()
        self.vae = vae
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False

        # Create the latent diffusion model
        self.model = LatentUNet(latent_dim=vae.latent_dim, time_emb_dim=128, hidden_dim=256)

        self.timesteps = timesteps
        self.lr = lr

        # Create a linear beta schedule
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def forward_diffusion(self, x0, t, noise):
        # q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*noise
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt().view(-1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.size(0)

        # Encode images to latent space
        with torch.no_grad():
            z = self.vae.encode(x)

        # Diffusion process in latent space
        t = torch.randint(0, self.timesteps, (batch_size,), device=z.device)
        noise = torch.randn_like(z)
        z_noisy = self.forward_diffusion(z, t, noise)
        noise_pred = self.model(z_noisy, t)

        # Loss is MSE between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, num_samples=16, ddim_steps=50):
        self.model.eval()
        device = next(self.model.parameters()).device

        # Start from pure Gaussian noise in latent space
        z = torch.randn(num_samples, self.vae.latent_dim, device=device)

        # Create a schedule for DDIM (reverse timesteps)
        ddim_timesteps = torch.linspace(self.timesteps - 1, 0, steps=ddim_steps, dtype=torch.long, device=device)

        for i in range(len(ddim_timesteps) - 1):
            t = ddim_timesteps[i]
            t_next = ddim_timesteps[i + 1]
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(z, t_batch)

            # DDIM sampling formulation
            alpha_bar_t = self.alpha_bar[t]
            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()

            # Predict x0
            z0_pred = (z - sqrt_one_minus_alpha_bar_t.view(-1, 1) * noise_pred) / sqrt_alpha_bar_t.view(-1, 1)
            alpha_bar_t_next = self.alpha_bar[t_next]
            sqrt_alpha_bar_t_next = alpha_bar_t_next.sqrt()

            # Compute z for the next timestep
            z = sqrt_alpha_bar_t_next.view(-1, 1) * z0_pred + (1 - alpha_bar_t_next).sqrt().view(-1, 1) * noise_pred

        # Decode the latent vectors to images
        return self.vae.decode(z)


def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t - 0.5) * 2)  # Scale images to [-1, 1]
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
