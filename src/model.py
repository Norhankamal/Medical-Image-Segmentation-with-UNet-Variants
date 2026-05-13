"""

UNet architecture for binary nuclei segmentation.

Based on: Ronneberger et al., MICCAI 2015
Adapted for binary segmentation on the 2018 Data Science Bowl dataset.

"""

import yaml
import torch
import torch.nn as nn


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class DoubleConv(nn.Module):
    """
    Basic building block of UNet.
    Applies two rounds of Conv2d -> BatchNorm -> ReLU.
    Used at every level in both encoder and decoder.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    UNet for binary image segmentation.

    Encoder compresses the image through 4 levels,
    doubling filters at each step (64->128->256->512).
    Bottleneck processes the most abstract representation (1024 filters).
    Decoder reconstructs spatial detail through 4 upsampling levels.
    Skip connections concatenate encoder outputs into the decoder
    to recover fine boundary detail lost during downsampling.
    Final 1x1 conv + sigmoid produces a binary probability mask.

    Adapted from the original paper:
        - Input size: 256x256 (paper used 572x572)
        - Same padding to preserve spatial dimensions
        - Sigmoid output for binary task (paper used softmax)
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        # Each level: upsample -> concat skip connection -> DoubleConv
        # DoubleConv input is doubled due to skip concatenation
        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output: 1x1 conv + sigmoid -> binary probability map
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.output_conv(d1))


if __name__ == "__main__":
    cfg   = load_config()
    model = UNet(
        in_channels  = cfg["model"]["in_channels"],
        out_channels = cfg["model"]["out_channels"]
    )

    dummy  = torch.randn(2, 3, 256, 256)
    output = model(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Output range : [{output.min():.3f}, {output.max():.3f}]")
    print(f"Total params : {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (2, 1, 256, 256), \
        f"Shape mismatch — expected (2,1,256,256), got {output.shape}"
    print("\nSanity check passed.")

    try:
        from torchsummary import summary
        summary(model, (3, 256, 256))
    except Exception as e:
        print(f"torchsummary skipped: {e}")