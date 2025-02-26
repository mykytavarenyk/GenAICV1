import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm


# -----------------------------
# DeepCache Implementation
# -----------------------------
class DeepCache:
    def __init__(self, model, cache_interval=4, cache_ratio=0.8):
        """
        Initialize DeepCache for a diffusion model.

        Args:
            model: The diffusion model.
            cache_interval: Number of steps between caching.
            cache_ratio: Interpolation ratio between cached and computed features.
        """
        self.model = model
        self.cache_interval = cache_interval
        self.cache_ratio = cache_ratio
        self.cached_features = {}

    def register_hooks(self):
        """Register forward hooks to store intermediate features."""
        self.hooks = []
        self.hook_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, UncondBlock):
                layer_id = len(self.hook_layers)
                self.hook_layers.append(module)
                hook = module.register_forward_hook(self.create_hook(layer_id))
                self.hooks.append(hook)

    def create_hook(self, layer_id):
        """Create a hook function for a specific layer."""

        def hook_fn(module, input_feat, output_feat):
            if self.should_cache:
                self.cached_features[layer_id] = output_feat.detach().clone()
            if self.should_blend and layer_id in self.cached_features:
                output_feat = self.cache_ratio * self.cached_features[layer_id] + (1 - self.cache_ratio) * output_feat
            return output_feat

        return hook_fn

    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def enable_caching(self):
        """Enable feature caching."""
        self.should_cache = True
        self.should_blend = False

    def enable_blending(self):
        """Enable feature blending."""
        self.should_cache = False
        self.should_blend = True

    def disable(self):
        """Disable both caching and blending."""
        self.should_cache = False
        self.should_blend = False

    def clear_cache(self):
        """Clear the cached features."""
        self.cached_features = {}


# -----------------------------
# Sinusoidal Positional Embedding for Time
# -----------------------------
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


# -----------------------------
# Residual Block with Time Conditioning (Unconditional)
# -----------------------------
class UncondBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.time_dense = nn.Linear(time_emb_dim, out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.relu(h)
        h = h + self.time_dense(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        return self.relu(h + self.shortcut(x))


# -----------------------------
# Downsample and Upsample Blocks
# -----------------------------
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


# -----------------------------
# Unconditional U-Net for Diffusion
# -----------------------------
class UnconditionalUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.block1 = UncondBlock(base_channels, base_channels, time_emb_dim)
        self.down1 = DownSample(base_channels, base_channels * 2)
        self.block2 = UncondBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.down2 = DownSample(base_channels * 2, base_channels * 4)
        self.block3 = UncondBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.bottleneck = UncondBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.up1 = UpSample(base_channels * 4, base_channels * 2)
        self.block4 = UncondBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.up2 = UpSample(base_channels * 2, base_channels)
        self.block5 = UncondBlock(base_channels, base_channels, time_emb_dim)
        self.conv_out = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        h = self.conv_in(x)
        h = self.block1(h, t_emb)
        skip1 = h
        h = self.down1(h)
        h = self.block2(h, t_emb)
        skip2 = h
        h = self.down2(h)
        h = self.block3(h, t_emb)
        h = self.bottleneck(h, t_emb)
        h = self.up1(h)
        h = h + skip2
        h = self.block4(h, t_emb)
        h = self.up2(h)
        h = h + skip1
        h = self.block5(h, t_emb)
        return self.conv_out(h)


# -----------------------------
# Diffusion Model with DDIM Sampling (Unconditional)
# -----------------------------
class DiffusionModel(pl.LightningModule):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, lr=2e-4):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.lr = lr
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def forward_diffusion(self, x0, t, noise):
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x.device)
        noise = torch.randn_like(x)
        x_noisy = self.forward_diffusion(x, t, noise)
        noise_pred = self.model(x_noisy, t)
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, num_samples=16, ddim_steps=50):
        self.model.eval()
        device = next(self.model.parameters()).device
        img_size = 28
        x = torch.randn(num_samples, 1, img_size, img_size, device=device)
        ddim_timesteps = torch.linspace(self.timesteps - 1, 0, steps=ddim_steps,
                                        dtype=torch.long, device=device)
        for i in range(len(ddim_timesteps) - 1):
            t = ddim_timesteps[i]
            t_next = ddim_timesteps[i + 1]
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = self.model(x, t_batch)
            alpha_bar_t = self.alpha_bar[t]
            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
            alpha_bar_t_next = self.alpha_bar[t_next]
            sqrt_alpha_bar_t_next = alpha_bar_t_next.sqrt()
            x = sqrt_alpha_bar_t_next * x0_pred + (1 - alpha_bar_t_next).sqrt() * noise_pred
        return x

    @torch.no_grad()
    def sample_with_deepcache(self, num_samples=16, ddim_steps=50, cache_interval=4, cache_ratio=0.8):
        """
        Generate samples using DDIM with DeepCache optimization.

        Args:
            num_samples: Number of samples to generate.
            ddim_steps: Total denoising steps.
            cache_interval: Interval between caching features.
            cache_ratio: Interpolation ratio between cached and computed features.

        Returns:
            Generated samples, total time, and per-step timing.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        img_size = 28

        deep_cache = DeepCache(self.model, cache_interval=cache_interval, cache_ratio=cache_ratio)
        deep_cache.register_hooks()

        x = torch.randn(num_samples, 1, img_size, img_size, device=device)
        ddim_timesteps = torch.linspace(self.timesteps - 1, 0, steps=ddim_steps,
                                        dtype=torch.long, device=device)
        start_time = time.time()
        step_times = []
        for i in range(len(ddim_timesteps) - 1):
            step_start = time.time()
            t = ddim_timesteps[i]
            t_next = ddim_timesteps[i + 1]
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            if i % cache_interval == 0:
                deep_cache.enable_caching()
                deep_cache.clear_cache()
            else:
                deep_cache.enable_blending()
            noise_pred = self.model(x, t_batch)
            deep_cache.disable()
            alpha_bar_t = self.alpha_bar[t]
            sqrt_alpha_bar_t = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
            alpha_bar_t_next = self.alpha_bar[t_next]
            sqrt_alpha_bar_t_next = alpha_bar_t_next.sqrt()
            x = sqrt_alpha_bar_t_next * x0_pred + (1 - alpha_bar_t_next).sqrt() * noise_pred
            step_times.append(time.time() - step_start)
        deep_cache.remove_hooks()
        total_time = time.time() - start_time
        return x, total_time, step_times


# -----------------------------
# Data Loading (MNIST)
# -----------------------------
def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t - 0.5) * 2)
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# -----------------------------
# Evaluation Metrics for Sample Quality
# -----------------------------
def compute_fid(real_images, generated_images):
    """
    Simplified FID calculation for MNIST.
    This is a basic approximation.
    """
    real_flat = real_images.reshape(real_images.shape[0], -1)
    gen_flat = generated_images.reshape(generated_images.shape[0], -1)
    real_mean = np.mean(real_flat, axis=0)
    gen_mean = np.mean(gen_flat, axis=0)
    real_cov = np.cov(real_flat, rowvar=False)
    gen_cov = np.cov(gen_flat, rowvar=False)
    mean_diff = np.sum((real_mean - gen_mean) ** 2)
    eps = 1e-6
    real_cov = real_cov + np.eye(real_cov.shape[0]) * eps
    gen_cov = gen_cov + np.eye(gen_cov.shape[0]) * eps
    covmean = np.sqrt(real_cov @ gen_cov)
    if np.isnan(covmean).any():
        return mean_diff
    tr_covmean = np.trace(covmean)
    fid = mean_diff + np.trace(real_cov) + np.trace(gen_cov) - 2 * tr_covmean
    return fid


def compute_mse(real_images, generated_images):
    """Compute Mean Squared Error between real and generated images."""
    return ((real_images - generated_images) ** 2).mean()


# -----------------------------
# DeepCache Experiments
# -----------------------------
def run_deepcache_experiments(diffusion_model, real_images, num_samples=16, ddim_steps=50):
    """Run experiments with different DeepCache configurations."""
    device = next(diffusion_model.parameters()).device
    cache_intervals = [1, 2, 4, 8, 16]
    cache_ratios = [0.2, 0.5, 0.8, 0.9]
    print("Running baseline (no caching)...")
    start_time = time.time()
    baseline_samples = diffusion_model.sample(num_samples=num_samples, ddim_steps=ddim_steps)
    baseline_time = time.time() - start_time
    baseline_np = baseline_samples.cpu().numpy()
    real_np = real_images.cpu().numpy()
    baseline_fid = compute_fid(real_np, baseline_np)
    baseline_mse = compute_mse(real_np, baseline_np)
    print(f"Baseline - Time: {baseline_time:.2f}s, FID: {baseline_fid:.2f}, MSE: {baseline_mse:.4f}")
    results = {
        'baseline': {
            'time': baseline_time,
            'fid': baseline_fid,
            'mse': baseline_mse,
            'samples': baseline_samples
        },
        'experiments': []
    }
    for interval in cache_intervals:
        for ratio in cache_ratios:
            print(f"Testing DeepCache with interval={interval}, ratio={ratio}...")
            samples, total_time, step_times = diffusion_model.sample_with_deepcache(
                num_samples=num_samples, ddim_steps=ddim_steps,
                cache_interval=interval, cache_ratio=ratio
            )
            samples_np = samples.cpu().numpy()
            fid = compute_fid(real_np, samples_np)
            mse = compute_mse(real_np, samples_np)
            speedup = baseline_time / total_time
            print(
                f"Interval={interval}, Ratio={ratio} - Time: {total_time:.2f}s (Speedup: {speedup:.2f}x), FID: {fid:.2f}, MSE: {mse:.4f}")
            results['experiments'].append({
                'interval': interval,
                'ratio': ratio,
                'time': total_time,
                'speedup': speedup,
                'fid': fid,
                'mse': mse,
                'samples': samples,
                'step_times': step_times
            })
    return results


# -----------------------------
# Visualization Functions
# -----------------------------
def make_grid(tensor, nrow=8):
    """Simple function to create a grid of images."""
    b, c, h, w = tensor.shape
    ncol = (b + nrow - 1) // nrow
    grid = torch.zeros(c, h * nrow, w * ncol, device=tensor.device)
    for idx, img in enumerate(tensor):
        i = idx % nrow
        j = idx // nrow
        grid[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = img
    return grid


def show_samples(samples, nrow=4, title=None):
    """Display generated samples in a grid."""
    samples = (samples + 1) / 2  # Scale from [-1, 1] to [0, 1]
    grid = make_grid(samples, nrow=nrow)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    if title:
        plt.title(title)


def plot_metrics(results):
    """Plot metrics from DeepCache experiments."""
    baseline_time = results['baseline']['time']
    baseline_fid = results['baseline']['fid']
    baseline_mse = results['baseline']['mse']
    intervals = []
    ratios = []
    speedups = []
    fids = []
    mses = []
    for exp in results['experiments']:
        intervals.append(exp['interval'])
        ratios.append(exp['ratio'])
        speedups.append(exp['speedup'])
        fids.append(exp['fid'])
        mses.append(exp['mse'])
    combinations = [f"I={i}, R={r:.1f}" for i, r in zip(intervals, ratios)]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.bar(combinations, speedups)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
    plt.title('Speedup vs Baseline')
    plt.xticks(rotation=45)
    plt.ylabel('Speedup Factor (×)')
    plt.tight_layout()
    plt.subplot(1, 3, 2)
    plt.bar(combinations, fids)
    plt.axhline(y=baseline_fid, color='r', linestyle='--', label='Baseline')
    plt.title('FID Score (lower is better)')
    plt.xticks(rotation=45)
    plt.ylabel('FID')
    plt.tight_layout()
    plt.subplot(1, 3, 3)
    plt.bar(combinations, mses)
    plt.axhline(y=baseline_mse, color='r', linestyle='--', label='Baseline')
    plt.title('MSE (lower is better)')
    plt.xticks(rotation=45)
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig('deepcache_metrics.png')
    quality_speedup_tradeoff = [(fid / baseline_fid) * (mse / baseline_mse) / speedup for fid, mse, speedup in
                                zip(fids, mses, speedups)]
    best_config_idx = np.argmin(quality_speedup_tradeoff)
    print(
        f"\nBest quality-speedup tradeoff: Interval={intervals[best_config_idx]}, Ratio={ratios[best_config_idx]:.1f}")
    print(
        f"  Speedup: {speedups[best_config_idx]:.2f}x, FID: {fids[best_config_idx]:.2f}, MSE: {mses[best_config_idx]:.4f}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    show_samples(results['baseline']['samples'], title="Baseline Samples")
    plt.subplot(1, 2, 2)
    show_samples(results['experiments'][best_config_idx]['samples'],
                 title=f"DeepCache I={intervals[best_config_idx]}, R={ratios[best_config_idx]:.1f}")
    plt.tight_layout()
    plt.savefig('deepcache_sample_comparison.png')
    return intervals[best_config_idx], ratios[best_config_idx]
