import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, h_dim=256, z_dim=128):
        super(ConvVAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(4 * 4 * 128, h_dim)
        self.fc_bn1 = nn.BatchNorm1d(h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)  # mu
        self.fc22 = nn.Linear(h_dim, z_dim)  # log_var

        # Decoder
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc_bn3 = nn.BatchNorm1d(h_dim)
        self.fc4 = nn.Linear(h_dim, 4 * 4 * 128)
        self.fc_bn4 = nn.BatchNorm1d(4 * 4 * 128)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()

        # Beta parameter for KL divergence weight in loss function
        self.beta = 0.1  # Initial beta value

    def encode(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 4 * 4 * 128)
        x = self.leaky_relu(self.fc_bn1(self.fc1(x)))
        mu = self.fc21(x)
        log_var = self.fc22(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x = self.leaky_relu(self.fc_bn3(self.fc3(z)))
        x = self.leaky_relu(self.fc_bn4(self.fc4(x)))
        x = x.view(-1, 128, 4, 4)
        x = self.leaky_relu(self.bn4(self.deconv1(x)))
        x = self.leaky_relu(self.bn5(self.deconv2(x)))
        x = self.tanh(self.deconv3(x))  # Output activation adjusted for normalized data
        return x  # Output shape: [B, 3, 32, 32]

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    def loss_function(self, outputs, x):
        recon_x, mu, log_var = outputs
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        # Total loss with beta parameter
        loss = recon_loss + self.beta * kl_loss
        return loss

    def generate_images(self, outputs=None, num_samples=16, device='cpu'):
        if outputs is not None:
            # Generate images from outputs (reconstructed images)
            recon_x, _, _ = outputs
            return recon_x
        else:
            # Generate images by sampling random latent vectors
            z = torch.randn(num_samples, self.fc21.out_features).to(device)
            generated_images = self.decode(z)
            return generated_images
