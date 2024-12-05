from models.discriminator import Discriminator
from models.generator import Generator
import torch.nn as nn

# Custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGAN(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, ndf=64, nc=3):
        super(DCGAN, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nc = nc

        self.netG = Generator(ngpu, nc, nz, ngf)
        self.netD = Discriminator(ngpu, nc, ndf)

        # Apply weights initialization
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # Loss function
        self.criterion = nn.BCELoss()

        # Fixed noise for generating images
        self.fixed_noise = torch.randn(64, nz, 1, 1)

    def forward(self, x):
        # GANs don't have a standard forward pass
        return None

    def loss_function(self, data, device):
        real_imgs = data[0].to(device)
        b_size = real_imgs.size(0)

        self.netD.zero_grad()
        label = torch.full((b_size,), 1., device=device)
        output_real = self.netD(real_imgs)
        errD_real = self.criterion(output_real, label)
        D_x = output_real.mean().item()

        noise = torch.randn(b_size, self.nz, 1, 1, device=device)
        fake_imgs = self.netG(noise)
        label.fill_(0.)
        output_fake = self.netD(fake_imgs.detach())
        errD_fake = self.criterion(output_fake, label)
        D_G_z1 = output_fake.mean().item()
        errD = errD_real + errD_fake
        errD.backward()
        self.optimizerD.step()

        self.netG.zero_grad()
        label.fill_(1.)
        output = self.netD(fake_imgs)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()

        losses = {
            'errD': errD.item(),
            'errG': errG.item(),
            'D_x': D_x,
            'D_G_z1': D_G_z1,
            'D_G_z2': D_G_z2
        }
        return losses, fake_imgs

    def generate_images(self, outputs=None, num_samples=64, device='cpu'):
        self.netG.eval()
        with torch.no_grad():
            fake_images = self.netG(self.fixed_noise.to(device)).cpu()
        return fake_images

    def set_optimizers(self, optimizerD, optimizerG):
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG


