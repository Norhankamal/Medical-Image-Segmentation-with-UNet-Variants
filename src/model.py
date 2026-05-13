#UNet architecture for binary medical image segmentation.

import yaml
import torch
import torch.nn as nn


def load_config(path="configs/config.yaml"):
    """Load hyperparameters from config.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


class DoubleConv(nn.Module):
    """
    Two consecutive Conv -> BatchNorm -> ReLU blocks.

    This is the core building block used at every level of UNet.
    Applied in both encoder and decoder paths.

    Why two convolutions?
        Each convolution learns different features. The first
        detects basic patterns, the second refines them.

    Why BatchNorm?
        Normalizes layer outputs — stabilizes and speeds training.

    Why ReLU?
        Adds non-linearity. Without it the entire network
        collapses into one linear transformation and learns nothing.

    Args:
        in_channels  : number of input feature maps
        out_channels : number of output feature maps
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Second convolution
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

    ENCODER:    shrinks image step by step, learns WHAT features
                exist. Filters double at each level: 64→128→256→512.
    
    BOTTLENECK: deepest point — most abstract understanding
                of the image. 1024 filters here.
    
    DECODER:    grows back to original size, learns WHERE features
                are. Filters halve at each level: 512→256→128→64.

    Skip connections:
        connect each encoder level directly to its
        matching decoder level. This passes fine spatial
        detail (edges, boundaries) that would otherwise
        be lost during downsampling.
        Without skip connections nucleus boundaries
        become blurry and imprecise.

    Args:
        in_channels  : number of input channels (3 for RGB)
        out_channels : number of output channels (1 for binary)
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────
        # Filters double at each level: 64 -> 128 -> 256 -> 512
        # MaxPool halves spatial size: 256->128->64->32->16
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Bottleneck ────────────────────────────────────────
        # Deepest point — most abstract representation
        self.bottleneck = DoubleConv(512, 1024)

        # ── Decoder ───────────────────────────────────────────
        # ConvTranspose2d doubles spatial size (reverse of MaxPool)
        # After upsampling, concatenate skip connection from encoder
        # Input to DoubleConv is doubled because of concatenation
        # Example: up4 output=512 + skip from enc4=512 = 1024 input
        self.up4  = nn.ConvTranspose2d(1024, 512,
                                        kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3  = nn.ConvTranspose2d(512, 256,
                                        kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2  = nn.ConvTranspose2d(256, 128,
                                        kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64,
                                        kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # ── Output ────────────────────────────────────────────
        # 1x1 conv collapses 64 channels to 1 probability map
        # Sigmoid squashes each pixel value to range [0, 1]
        # Pixel > 0.5 = nucleus, Pixel < 0.5 = background
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder — save each output for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder — upsample then concatenate skip connection
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Output — sigmoid produces probability map in [0, 1]
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
    except ImportError:
        print("Run: pip install torchsummary for detailed summary")