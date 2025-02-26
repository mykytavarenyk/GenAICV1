import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Set random seed for reproducibility
pl.seed_everything(42)

BATCH_SIZE = 64
NUM_EPOCHS = 30
LATENT_DIM = 64
IMAGE_SIZE = 28
LOG_IMAGES_EVERY_N_STEPS = 100


class CannyEdgeMNIST(Dataset):
    def __init__(self, root_dir, train=True, transform=None, download=True, low_threshold=50, high_threshold=150):
        self.mnist = MNIST(root=root_dir, train=train, transform=transform, download=download)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        # Convert to numpy for Canny processing
        img_np = (img.squeeze().numpy() * 255).astype(np.uint8)
        # Apply Canny edge detection
        edge = cv2.Canny(img_np, self.low_threshold, self.high_threshold)
        # Convert back to tensor and normalize to [0,1]
        edge_tensor = torch.from_numpy(edge).float() / 255.0
        edge_tensor = edge_tensor.unsqueeze(0)
        return img, edge_tensor, label


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return self.pool(x), x  # Return pooled and pre-pooled features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, skip):
        x = self.upconv(x)
        # Ensure dimensions match before concatenation
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class ZeroConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Initialize weights and biases to zero
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(1, 32, use_bn=False)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # Decoder
        self.dec1 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec3 = DecoderBlock(64, 32)
        # Output layer
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, control_features=None):
        # Encoder path
        p1, s1 = self.enc1(x)
        p2, s2 = self.enc2(p1)
        p3, s3 = self.enc3(p2)
        # Bridge
        b = self.bridge(p3)
        # Apply control features if provided
        if control_features is not None:
            c_s1, c_s2, c_s3, c_b = control_features
            s1 = s1 + c_s1
            s2 = s2 + c_s2
            s3 = s3 + c_s3
            b = b + c_b
        # Decoder path
        d1 = self.dec1(b, s3)
        d2 = self.dec2(d1, s2)
        d3 = self.dec3(d2, s1)
        # Output image
        return torch.sigmoid(self.output(d3))

class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (same structure as UNet encoder)
        self.enc1 = EncoderBlock(1, 32, use_bn=False)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # Zero convolutions for output control features
        self.zero_s1 = ZeroConv(32, 32)
        self.zero_s2 = ZeroConv(64, 64)
        self.zero_s3 = ZeroConv(128, 128)
        self.zero_b = ZeroConv(256, 256)

    def forward(self, control):
        p1, s1 = self.enc1(control)
        p2, s2 = self.enc2(p1)
        p3, s3 = self.enc3(p2)
        b = self.bridge(p3)
        c_s1 = self.zero_s1(s1)
        c_s2 = self.zero_s2(s2)
        c_s3 = self.zero_s3(s3)
        c_b = self.zero_b(b)
        return (c_s1, c_s2, c_s3, c_b)


class ControlNetMNIST(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.unet = UNet()
        self.controlnet = ControlNet()
        self.lr = lr

    def forward(self, img, control):
        control_features = self.controlnet(control)
        return self.unet(img, control_features)

    def training_step(self, batch, batch_idx):
        img, edge, _ = batch
        zeros = torch.zeros_like(img)
        generated = self(zeros, edge)
        loss = F.l1_loss(generated, img)
        self.log('train_loss', loss)
        # Log images every LOG_IMAGES_EVERY_N_STEPS
        if batch_idx % LOG_IMAGES_EVERY_N_STEPS == 0:
            n_examples = min(5, img.size(0))
            fig, axs = plt.subplots(n_examples, 3, figsize=(12, 2 * n_examples))
            for i in range(n_examples):
                axs[i, 0].imshow(img[i].cpu().squeeze(), cmap='gray')
                axs[i, 0].set_title('Original')
                axs[i, 0].axis('off')
                axs[i, 1].imshow(edge[i].cpu().squeeze(), cmap='gray')
                axs[i, 1].set_title('Canny Edge')
                axs[i, 1].axis('off')
                axs[i, 2].imshow(generated[i].detach().cpu().squeeze(), cmap='gray')
                axs[i, 2].set_title('Generated')
                axs[i, 2].axis('off')
            plt.tight_layout()
            plt.close(fig)
            tensorboard = self.logger.experiment
            tensorboard.add_figure('train_samples', fig, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        img, edge, _ = batch
        zeros = torch.zeros_like(img)
        generated = self(zeros, edge)
        val_loss = F.l1_loss(generated, img)
        self.log('val_loss', val_loss)
        if batch_idx == 0:
            n_examples = min(5, img.size(0))
            fig, axs = plt.subplots(n_examples, 3, figsize=(12, 2 * n_examples))
            for i in range(n_examples):
                axs[i, 0].imshow(img[i].cpu().squeeze(), cmap='gray')
                axs[i, 0].set_title('Original')
                axs[i, 0].axis('off')
                axs[i, 1].imshow(edge[i].cpu().squeeze(), cmap='gray')
                axs[i, 1].set_title('Canny Edge')
                axs[i, 1].axis('off')
                axs[i, 2].imshow(generated[i].detach().cpu().squeeze(), cmap='gray')
                axs[i, 2].set_title('Generated')
                axs[i, 2].axis('off')
            plt.tight_layout()
            plt.close(fig)
            tensorboard = self.logger.experiment
            tensorboard.add_figure('val_samples', fig, self.global_step)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)