import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

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

class UncondBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.time_dense = nn.Linear(time_emb_dim, out_channels)
        # If in/out channel dimensions differ, use a projection shortcut.
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.relu(h)
        # Inject the time embedding (broadcast spatially).
        h = h + self.time_dense(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        return self.relu(h + self.shortcut(x))

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim, emb_dim):
        super().__init__()
        self.query = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key = nn.Linear(emb_dim, feature_dim)
        self.value = nn.Linear(emb_dim, feature_dim)
        self.scale = feature_dim ** -0.5
        self.proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, x, emb):
        # x: (B, feature_dim, H, W), emb: (B, emb_dim)
        B, C, H, W = x.shape
        q = self.query(x).view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        k = self.key(emb).unsqueeze(1)  # (B, 1, C)
        v = self.value(emb).unsqueeze(1)  # (B, 1, C)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H*W, 1)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        out = self.proj(out)
        return x + out

class CombinedConditioningUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128,
                 num_classes=10, class_emb_dim=32, attn_emb_dim=64):

        super().__init__()
        self.num_classes = num_classes
        # Embedding for input-channel conditioning.
        self.class_emb_input = nn.Embedding(num_classes, class_emb_dim)
        # Separate embedding for cross-attention conditioning.
        self.class_emb_attn = nn.Embedding(num_classes, attn_emb_dim)
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)

        # Increase input channels to include class condition.
        self.conv_in = nn.Conv2d(in_channels + class_emb_dim, base_channels, kernel_size=3, padding=1)

        # Downsampling path.
        self.block1 = UncondBlock(base_channels, base_channels, time_emb_dim)
        self.down1 = DownSample(base_channels, base_channels * 2)
        self.block2 = UncondBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.down2 = DownSample(base_channels * 2, base_channels * 4)
        self.block3 = UncondBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Bottleneck.
        self.bottleneck = UncondBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        # Cross-attention block applied in the bottleneck.
        self.cross_attn = CrossAttentionBlock(feature_dim=base_channels * 4, emb_dim=attn_emb_dim)

        # Upsampling path.
        self.up1 = UpSample(base_channels * 4, base_channels * 2)
        self.block4 = UncondBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.up2 = UpSample(base_channels * 2, base_channels)
        self.block5 = UncondBlock(base_channels, base_channels, time_emb_dim)

        # Final convolution to predict noise.
        self.conv_out = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t, y):

        B, _, H, W = x.shape
        # Input-Channel Conditioning:
        # For dropped conditions (y < 0) use a zero embedding.
        mask = (y >= 0).float().unsqueeze(1)
        y_idx = y.clone()
        y_idx[y_idx < 0] = 0  # Clamp negative indices for lookup.
        class_emb_in = self.class_emb_input(y_idx) * mask  # (B, class_emb_dim)
        class_emb_in = class_emb_in.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        # Concatenate the input image with the spatially replicated class embedding.
        x = torch.cat([x, class_emb_in], dim=1)  # Now channels = in_channels + class_emb_dim

        # Initial convolution.
        h = self.conv_in(x)
        t_emb = self.time_embedding(t)  # (B, time_emb_dim)
        h = self.block1(h, t_emb)
        skip1 = h
        h = self.down1(h)
        h = self.block2(h, t_emb)
        skip2 = h
        h = self.down2(h)
        h = self.block3(h, t_emb)

        # Bottleneck.
        h = self.bottleneck(h, t_emb)

        # Cross-Attention Conditioning:
        mask_attn = (y >= 0).float().unsqueeze(1)
        y_idx_attn = y.clone()
        y_idx_attn[y_idx_attn < 0] = 0
        class_emb_attn = self.class_emb_attn(y_idx_attn) * mask_attn  # (B, attn_emb_dim)
        h = self.cross_attn(h, class_emb_attn)

        # Upsampling with skip connections.
        h = self.up1(h)
        h = h + skip2
        h = self.block4(h, t_emb)
        h = self.up2(h)
        h = h + skip1
        h = self.block5(h, t_emb)
        return self.conv_out(h)

# -----------------------------
# Diffusion Model with DDIM Sampling and Classifier-Free Guidance
# -----------------------------
class DiffusionModel(pl.LightningModule):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, lr=2e-4, drop_prob=0.1):

        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.lr = lr
        self.drop_prob = drop_prob

        # Create a linear beta schedule.
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def forward(self, x, t, y):
        return self.model(x, t, y)

    def forward_diffusion(self, x0, t, noise):
        # q(x_t | x0) = sqrt(alpha_bar_t)*x0 + sqrt(1 - alpha_bar_t)*noise
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def training_step(self, batch, batch_idx):
        x, y = batch  # MNIST returns (image, label)
        batch_size = x.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x.device)
        noise = torch.randn_like(x)
        x_noisy = self.forward_diffusion(x, t, noise)
        # With probability drop_prob, drop the condition (set y to -1).
        drop_mask = torch.rand(batch_size, device=x.device) < self.drop_prob
        y_dropped = y.clone()
        y_dropped[drop_mask] = -1
        noise_pred = self.model(x_noisy, t, y_dropped)
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, num_samples=16, ddim_steps=50, desired_class=0, guidance_scale=5.0):
        self.model.eval()
        device = next(self.model.parameters()).device
        img_size = 28
        # Start from pure Gaussian noise.
        x = torch.randn(num_samples, 1, img_size, img_size, device=device)
        ddim_timesteps = torch.linspace(self.timesteps - 1, 0, steps=ddim_steps, dtype=torch.long, device=device)

        for i in range(len(ddim_timesteps) - 1):
            t = ddim_timesteps[i]
            t_next = ddim_timesteps[i + 1]
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            # Conditional prediction.
            y_cond = torch.full((num_samples,), desired_class, device=device, dtype=torch.long)
            # Unconditional prediction (dropped condition).
            y_uncond = torch.full((num_samples,), -1, device=device, dtype=torch.long)
            eps_cond = self.model(x, t_batch, y_cond)
            eps_uncond = self.model(x, t_batch, y_uncond)
            # Classifier-free guidance:
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            alpha_bar_t = self.alpha_bar[t]
            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
            alpha_bar_t_next = self.alpha_bar[t_next]
            sqrt_alpha_bar_t_next = alpha_bar_t_next.sqrt()
            x = sqrt_alpha_bar_t_next * x0_pred + (1 - alpha_bar_t_next).sqrt() * eps
        return x
