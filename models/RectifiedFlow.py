import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256, hidden_dims=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.time_dim = time_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_channels = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(),
                    nn.MaxPool2d(2)
                )
            )
            in_channels = hidden_dim

        # Middle block
        self.middle_block = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.SiLU(),
        )

        # Time embeddings for each layer
        self.time_embeddings = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.time_embeddings.append(
                nn.Sequential(
                    nn.Linear(time_dim, hidden_dim),
                    nn.SiLU()
                )
            )

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_hidden_dims) - 1):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_hidden_dims[i], reversed_hidden_dims[i + 1],
                                       kernel_size=2, stride=2),
                    nn.Conv2d(reversed_hidden_dims[i + 1] * 2, reversed_hidden_dims[i + 1],
                              kernel_size=3, padding=1),
                    nn.BatchNorm2d(reversed_hidden_dims[i + 1]),
                    nn.SiLU(),
                    nn.Conv2d(reversed_hidden_dims[i + 1], reversed_hidden_dims[i + 1],
                              kernel_size=3, padding=1),
                    nn.BatchNorm2d(reversed_hidden_dims[i + 1]),
                    nn.SiLU(),
                )
            )

        # Final layer
        self.final_conv = nn.Conv2d(hidden_dims[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1)
        time_emb = self.time_mlp(t)

        # Initial convolution
        x = self.initial_conv(x)
        skips = [x]

        # Apply time embedding to the initial layer
        x = x + self.time_embeddings[0](time_emb).unsqueeze(-1).unsqueeze(-1)

        # Downsampling with time conditioning
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            x = x + self.time_embeddings[i + 1](time_emb).unsqueeze(-1).unsqueeze(-1)
            skips.append(x)

        # Middle block
        x = self.middle_block(x)

        # Upsampling with skip connections (reverse skip order)
        skips = skips[:-1]
        skips.reverse()
        for i, block in enumerate(self.up_blocks):
            x = block[0](x)  # Transposed conv for upsampling
            if x.shape != skips[i].shape:
                x = F.interpolate(x, size=skips[i].shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skips[i]], dim=1)
            for j in range(1, len(block)):
                x = block[j](x)
        return self.final_conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )
        self.feature_size = 256 * 2 * 2
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_var = nn.Linear(self.feature_size, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        att = self.attention(x)
        x = x * att
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, log_var


# -----------------------------
# Decoder for Latent-Based Diffusion Model
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=1, img_size=28):
        super(Decoder, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        self.bn_fc = nn.BatchNorm1d(256 * 2 * 2)
        self.act_fc = nn.LeakyReLU(0.2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        z = self.fc(z)
        z = self.bn_fc(z)
        z = self.act_fc(z)
        z = z.view(-1, 256, 2, 2)
        z = self.deconv1(z)
        z = self.deconv2(z)
        z = self.deconv3(z)
        x = self.final(z)
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return x


# -----------------------------
# Latent Diffusion MLP
# -----------------------------
class LatentMLP(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=512):
        super(LatentMLP, self).__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        x_t = torch.cat([x, t_embed], dim=-1)
        return self.net(x_t)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return x + 0.1 * self.block(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return embeddings


class RectifiedFlowDiffusion(pl.LightningModule):
    def __init__(self, img_size=28, batch_size=64, lr=2e-4):
        super(RectifiedFlowDiffusion, self).__init__()
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.model = UNet(in_channels=1, out_channels=1)
        self.save_hyperparameters()

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        t = torch.rand(imgs.shape[0], device=self.device)
        z = torch.randn_like(imgs)
        x_t = (1 - t.view(-1, 1, 1, 1)) * imgs + t.view(-1, 1, 1, 1) * z
        v_true = z - imgs
        v_pred = self.model(x_t, t)
        loss = F.mse_loss(v_pred, v_true)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        t = torch.rand(imgs.shape[0], device=self.device)
        z = torch.randn_like(imgs)
        x_t = (1 - t.view(-1, 1, 1, 1)) * imgs + t.view(-1, 1, 1, 1) * z
        v_true = z - imgs
        v_pred = self.model(x_t, t)
        loss = F.mse_loss(v_pred, v_true)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def generate_samples(self, num_samples=16, steps=100):
        self.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, 1, self.img_size, self.img_size, device=self.device)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.ones(num_samples, device=self.device) * (1.0 - i * dt)
                v = self.model(x, t)
                x = x - v * dt
            x = torch.clamp(x, 0, 1)
        return x

    def on_train_epoch_end(self):
        if self.current_epoch % 5 == 0:
            samples = self.generate_samples(num_samples=16)
            grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
            self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


class LatentRectifiedFlowDiffusion(pl.LightningModule):
    def __init__(self, latent_dim=128, img_size=28, batch_size=64, lr=1e-4, kl_weight=0.001, scheduler_gamma=0.95):
        super(LatentRectifiedFlowDiffusion, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.kl_weight = kl_weight
        self.scheduler_gamma = scheduler_gamma

        self.encoder = Encoder(in_channels=1, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=1, img_size=img_size)
        self.diffusion = LatentMLP(latent_dim=latent_dim, hidden_dim=512)
        self.save_hyperparameters()

        self.train_recon_loss_avg = 0.0
        self.train_diffusion_loss_avg = 0.0
        self.train_kl_loss_avg = 0.0
        self.val_recon_loss_avg = 0.0
        self.val_diffusion_loss_avg = 0.0
        self.val_kl_loss_avg = 0.0

    def forward(self, x):
        z, _, _ = self.encoder(x)
        return self.decoder(z)

    def _common_step(self, batch, batch_idx, stage):
        imgs, _ = batch
        z, mu, log_var = self.encoder(imgs)
        t = torch.rand(imgs.shape[0], device=self.device)
        noise = torch.randn_like(z)
        z_t = (1 - t.view(-1, 1)) * z + t.view(-1, 1) * noise
        v_true = noise - z
        v_pred = self.diffusion(z_t, t)
        mse_loss = F.mse_loss(v_pred, v_true)
        cos_sim = F.cosine_similarity(v_pred, v_true).mean()
        diffusion_loss = mse_loss - 0.1 * cos_sim
        x_recon = self.decoder(z)
        pixel_loss = F.mse_loss(x_recon, imgs)
        edge_loss = self._edge_loss(x_recon, imgs)
        recon_loss = pixel_loss + 0.1 * edge_loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)
        loss = diffusion_loss + recon_loss + self.kl_weight * kl_loss

        if stage == 'train':
            self.train_recon_loss_avg = 0.9 * self.train_recon_loss_avg + 0.1 * recon_loss.item()
            self.train_diffusion_loss_avg = 0.9 * self.train_diffusion_loss_avg + 0.1 * diffusion_loss.item()
            self.train_kl_loss_avg = 0.9 * self.train_kl_loss_avg + 0.1 * kl_loss.item()
            self.log('train_loss', loss)
            self.log('train_diffusion_loss', diffusion_loss)
            self.log('train_recon_loss', recon_loss)
            self.log('train_kl_loss', kl_loss)
            self.log('train_cos_sim', cos_sim)
        else:
            self.val_recon_loss_avg = 0.9 * self.val_recon_loss_avg + 0.1 * recon_loss.item()
            self.val_diffusion_loss_avg = 0.9 * self.val_diffusion_loss_avg + 0.1 * diffusion_loss.item()
            self.val_kl_loss_avg = 0.9 * self.val_kl_loss_avg + 0.1 * kl_loss.item()
            self.log('val_loss', loss)
            self.log('val_diffusion_loss', diffusion_loss)
            self.log('val_recon_loss', recon_loss)
            self.log('val_kl_loss', kl_loss)
            self.log('val_cos_sim', cos_sim)

        return loss

    def _edge_loss(self, x, y):
        def sobel_edge(img):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            edges_x = F.conv2d(img, sobel_x, padding=1)
            edges_y = F.conv2d(img, sobel_y, padding=1)
            return torch.sqrt(edges_x.pow(2) + edges_y.pow(2) + 1e-8)

        edges_x = sobel_edge(x)
        edges_y = sobel_edge(y)
        return F.mse_loss(edges_x, edges_y)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.scheduler_gamma
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def generate_samples(self, num_samples=16, steps=100):
        self.eval()
        with torch.no_grad():
            latent_noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.ones(num_samples, device=self.device) * (1.0 - i * dt)
                v1 = self.diffusion(latent_noise, t)
                latent_pred = latent_noise - v1 * dt
                if i < steps - 1:
                    t_next = torch.ones(num_samples, device=self.device) * (1.0 - (i + 1) * dt)
                    v2 = self.diffusion(latent_pred, t_next)
                    latent_noise = latent_noise - 0.5 * dt * (v1 + v2)
                else:
                    latent_noise = latent_pred
            samples = self.decoder(latent_noise)
        return samples

    def on_train_epoch_end(self):
        if self.current_epoch % 5 == 0:
            samples = self.generate_samples(num_samples=16)
            grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
            self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.mnist_train = torchvision.datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_val = torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)
