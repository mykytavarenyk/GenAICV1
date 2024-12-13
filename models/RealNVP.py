import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable, Optional, Dict
from enum import IntEnum

class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1

class Rescale(nn.Module):
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        return self.weight * x

class WNConv2d(nn.Module):
    """Weight-normalized 2D convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = WNConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_conv = WNConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip = x
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)
        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)
        x = x + skip
        return x

class ResNetModule(nn.Module):
    """ResNet for scale and translate factors in Real NVP."""
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        num_blocks,
        kernel_size,
        padding,
        double_after_norm,
    ):
        super(ResNetModule, self).__init__()
        self.in_norm = nn.BatchNorm2d(in_channels)
        self.double_after_norm = double_after_norm
        self.in_conv = WNConv2d(2 * in_channels, mid_channels, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)

        self.blocks = nn.ModuleList(
            [ResidualBlock(mid_channels, mid_channels) for _ in range(num_blocks)]
        )
        self.skips = nn.ModuleList(
            [WNConv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True) for _ in range(num_blocks)]
        )

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = WNConv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.in_norm(x)
        if self.double_after_norm:
            x *= 2.0
        x = torch.cat((x, -x), dim=1)
        x = F.relu(x)
        x = self.in_conv(x)
        x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)

        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_conv(x)

        return x

def checkerboard_mask(height, width, num_channels, reverse=False, device='cpu'):
    """
    Generates a checkerboard mask with the specified number of channels.

    Args:
        height (int): Height of the mask.
        width (int): Width of the mask.
        num_channels (int): Number of channels to repeat the mask across.
        reverse (bool): If True, invert the mask.
        device (str): Device to place the mask tensor.

    Returns:
        torch.Tensor: Checkerboard mask tensor of shape (1, num_channels, height, width).
    """
    checkerboard = [[(i + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=torch.float32, device=device)
    if reverse:
        mask = 1 - mask
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    mask = mask.repeat(1, num_channels, 1, 1)  # Repeat for the required number of channels
    return mask

class CouplingLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask):
        super(CouplingLayer, self).__init__()
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        if self.mask_type == MaskType.CHANNEL_WISE:
            in_channels = in_channels // 2

        self.st_net = ResNetModule(
            in_channels,
            mid_channels,
            2 * in_channels,
            num_blocks=num_blocks,
            kernel_size=3,
            padding=1,
            double_after_norm=(self.mask_type == MaskType.CHECKERBOARD),
        )

        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def forward(self, x, sldj=None, reverse=False):
        if self.mask_type == MaskType.CHECKERBOARD:
            num_channels = x.size(1)  # Dynamic channel size
            b = checkerboard_mask(x.size(2), x.size(3), num_channels, self.reverse_mask, device=x.device)
            x_b = x * b
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            s = s * (1 - b)
            t = t * (1 - b)

            if reverse:
                x = x * torch.exp(-s) - t
            else:
                x = (x + t) * torch.exp(s)
                sldj += s.view(s.size(0), -1).sum(-1)
        else:
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            st = self.st_net(x_id)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))

            if reverse:
                x_change = x_change * torch.exp(-s) - t
            else:
                x_change = (x_change + t) * torch.exp(s)
                sldj += s.view(s.size(0), -1).sum(-1)

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x, sldj

def squeeze(x):
    b, c, h, w = x.size()
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(b, c * 4, h // 2, w // 2)
    return x

def unsqueeze(x):
    b, c, h, w = x.size()
    x = x.view(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(b, c // 4, h * 2, w * 2)
    return x

class _RealNVP(nn.Module):
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_RealNVP, self).__init__()
        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
        ])

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True)
            )
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
            ])
            self.next_block = _RealNVP(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            if not self.is_last_block:
                x = squeeze(x)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = unsqueeze(x)

                x = squeeze(x)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = unsqueeze(x)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:
                x = squeeze(x)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = unsqueeze(x)

                x = squeeze(x)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = unsqueeze(x)
        return x, sldj

class RealNVP(nn.Module):
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(RealNVP, self).__init__()
        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32))
        self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

    def forward(self, x, reverse=False):
        sldj = x.new_zeros(x.size(0))
        if not reverse:
            if x.min() < 0 or x.max() > 1:
                raise ValueError(
                    f'Expected x in [0, 1], got x with min/max {x.min().item()}/{x.max().item()}'
                )
            x, sldj = self._pre_process(x, sldj)

        x, sldj = self.flows(x, sldj, reverse)

        if reverse:
            x = torch.sigmoid(x)
        return x, sldj

    def _pre_process(self, x, sldj):
        y = (x * 255.0 + torch.rand_like(x)) / 256.0
        y = (2 * y - 1) * self.data_constraint
        y = (y + 1) / 2
        y = y.log() - (1.0 - y).log()

        ldj = F.softplus(y) + F.softplus(-y) - F.softplus(
            (1.0 - self.data_constraint).log() - self.data_constraint.log()
        )
        sldj += ldj.view(ldj.size(0), -1).sum(-1)

        return y, sldj

    def generate_images(self, num_samples=16, device='cpu'):
        """Generates images by sampling from the latent space."""
        with torch.no_grad():
            z = torch.randn(num_samples, 3, 32, 32).to(device)
            samples, _ = self(z, reverse=True)
            samples = torch.sigmoid(samples)
        return samples

    def loss_function(self, outputs, inputs):
        """Computes the RealNVP loss."""
        # Assuming outputs are z and sldj
        z, sldj = outputs
        return RealNVPLoss()(z, sldj)

class RealNVPLoss(nn.Module):
    def __init__(self, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        # Prior log-likelihood
        prior_ll = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * np.pi, device=z.device)))
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1) - torch.log(torch.tensor(self.k, device=z.device)) * z[0].numel()
        ll = prior_ll + sldj
        nll = -ll.mean()
        return nll