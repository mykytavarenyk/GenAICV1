import torch.nn as nn

# Adjusted Discriminator class for 32x32 images
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input: (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # Output: (ndf) x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # Output: (ndf*2) x 8 x 8
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # Output: (ndf*4) x 4 x 4
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Final layer
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),  # Output: (1) x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)