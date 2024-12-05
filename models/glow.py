import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable, Optional, Dict
from enum import IntEnum

class PadChannel(object):
    """
    Pads the channel dimension of an image tensor to a specified number of channels.
    """

    def __init__(self, target_channels=4):
        self.target_channels = target_channels

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Image tensor of shape [C, H, W]

        Returns:
            Tensor: Padded image tensor of shape [target_channels, H, W]
        """
        current_channels = tensor.size(0)
        if current_channels < self.target_channels:
            padding = torch.zeros(self.target_channels - current_channels, tensor.size(1), tensor.size(2))
            tensor = torch.cat([tensor, padding], dim=0)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(target_channels={self.target_channels})'


# --- [Glow Model and Associated Classes] ---
class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class ActNorm(nn.Module):
    def __init__(self, num_channels, device='cpu'):
        super(ActNorm, self).__init__()
        self.num_channels = num_channels
        self.initialized = False
        self.device = device

        # Initialize scale and bias parameters
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x, reverse=False):
        if not self.initialized:
            with torch.no_grad():
                # Compute mean and std over batch, height, width
                mean = x.mean(dim=[0, 2, 3], keepdim=True)
                std = x.std(dim=[0, 2, 3], keepdim=True)
                self.bias.data = -mean
                self.scale.data = 1 / (std + 1e-6)
                self.initialized = True

        if reverse:
            x = (x - self.bias) * self.scale
        else:
            x = x * self.scale + self.bias
        # Compute log determinant
        log_det_J = torch.sum(torch.log(torch.abs(self.scale)))
        log_det_J = log_det_J.expand(x.size(0))
        return x, log_det_J


class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super(Invertible1x1Conv, self).__init__()
        # Initialize a random orthogonal matrix
        w_init = np.linalg.qr(np.random.randn(num_channels, num_channels))[0].astype(np.float32)
        w_init = torch.from_numpy(w_init)
        self.weight = nn.Parameter(w_init)

    def forward(self, x, reverse=False):
        batch_size, num_channels, height, width = x.size()
        if reverse:
            w = torch.inverse(self.weight)
        else:
            w = self.weight
        # Reshape for convolution
        w = w.view(num_channels, num_channels, 1, 1)
        x = F.conv2d(x, w)
        # Compute log determinant
        log_det_J = torch.log(torch.abs(torch.det(self.weight)))
        log_det_J = log_det_J.expand(batch_size)
        return x, log_det_J


class AffineCoupling(nn.Module):
    def __init__(self, num_channels, hidden_channels):
        super(AffineCoupling, self).__init__()
        self.num_channels = num_channels
        self.hidden_channels = hidden_channels

        # Neural network to compute scale and translation
        self.net = nn.Sequential(
            nn.Conv2d(num_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, reverse=False):
        x_a, x_b = x.chunk(2, dim=1)
        s_t = self.net(x_a)
        s, t = s_t.chunk(2, dim=1)
        s = torch.sigmoid(s + 2)  # Scale factor to keep gradients stable
        if reverse:
            x_a = x_a
            x_b = (x_b - t) / s
        else:
            x_a = x_a
            x_b = x_b * s + t
        x = torch.cat([x_a, x_b], dim=1)
        # Compute log determinant of the Jacobian
        log_det_J = torch.sum(torch.log(s.view(s.size(0), -1)), dim=1)
        return x, log_det_J


class GlowBlock(nn.Module):
    def __init__(self, num_channels, hidden_channels):
        super(GlowBlock, self).__init__()
        self.actnorm = ActNorm(num_channels)
        self.inv1x1 = Invertible1x1Conv(num_channels)
        self.affine_coupling = AffineCoupling(num_channels, hidden_channels)

    def forward(self, x, reverse=False):
        log_det_J_total = 0
        # ActNorm
        x, log_det_J = self.actnorm(x, reverse)
        log_det_J_total += log_det_J
        # Invertible 1x1 Conv
        x, log_det_J = self.inv1x1(x, reverse)
        log_det_J_total += log_det_J
        # Affine Coupling
        x, log_det_J = self.affine_coupling(x, reverse)
        log_det_J_total += log_det_J
        return x, log_det_J_total


class SqueezeLayer(nn.Module):
    def __init__(self):
        super(SqueezeLayer, self).__init__()

    def forward(self, x, reverse=False):
        if reverse:
            x = unsqueeze(x)
        else:
            x = squeeze(x)
        # Squeeze doesn't change the log determinant
        log_det_J = torch.zeros(x.size(0), device=x.device)
        return x, log_det_J


def squeeze(x):
    b, c, h, w = x.size()
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("Height and Width must be divisible by 2 for squeezing.")
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(b, c * 4, h // 2, w // 2)
    return x


def unsqueeze(x):
    b, c, h, w = x.size()
    if c % 4 != 0:
        raise ValueError("Number of channels must be divisible by 4 for unsqueezing.")
    x = x.view(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(b, c // 4, h * 2, w * 2)
    return x


class Glow(nn.Module):
    def __init__(self, num_scales, num_blocks, num_channels, hidden_channels, device='cpu'):
        super(Glow, self).__init__()
        self.device = device
        self.num_scales = num_scales
        self.num_blocks = num_blocks
        self.hidden_channels = hidden_channels

        self.flows = nn.ModuleList()
        self.current_num_channels = num_channels  # Track current number of channels
        self.current_height = 32  # Initial image height
        self.current_width = 32  # Initial image width

        for scale in range(num_scales):
            for block in range(num_blocks):
                self.flows.append(GlowBlock(self.current_num_channels, hidden_channels))
            if scale < num_scales - 1:
                self.flows.append(SqueezeLayer())
                self.current_num_channels *= 4  # Update number of channels after squeezing
                self.current_height = self.current_height // 2
                self.current_width = self.current_width // 2

        self.final_num_channels = self.current_num_channels  # For image generation
        self.final_height = self.current_height
        self.final_width = self.current_width

    def forward(self, x, reverse=False):
        log_det_J_total = 0
        if reverse:
            for flow in reversed(self.flows):
                x, log_det_J = flow(x, reverse)
                log_det_J_total += log_det_J
        else:
            for flow in self.flows:
                x, log_det_J = flow(x, reverse)
                log_det_J_total += log_det_J
        return x, log_det_J_total

    def generate_images(self, num_samples=16, device='cpu'):
        """Generates images by sampling from the latent space."""
        with torch.no_grad():
            # Initialize z with standard normal, matching the final number of channels and spatial size
            z = torch.randn(num_samples, self.final_num_channels, self.final_height, self.final_width).to(device)
            # Pass through reverse flows
            for flow in reversed(self.flows):
                z, _ = flow(z, reverse=True)
            # Apply sigmoid to get pixel values in [0,1]
            samples = torch.sigmoid(z)
            # Remove the extra padded channel to restore 3 channels
            samples = samples[:, :3, :, :]
        return samples

    def loss_function(self, outputs, inputs):
        """Computes the Glow loss."""
        z, log_det_J = outputs
        # Compute log-likelihood under standard normal prior
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi))
        log_pz = log_pz.view(z.size(0), -1).sum(dim=1)
        # Total log-likelihood
        log_px = log_pz + log_det_J
        # Negative log-likelihood loss
        nll = -log_px.mean()
        return nll

