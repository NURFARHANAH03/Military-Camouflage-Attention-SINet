import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -----------------------
# Double Conv Block (Decoder only)
# -----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------
# U-Net with Pretrained ResNet-50 Encoder
# -----------------------
class UNetResNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(UNetResNet, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        encoder_channels = [64, 256, 512, 1024, 2048]

        # ---------------- Encoder ----------------
        self.inconv = nn.Sequential(
            resnet.conv1,   # 7x7 conv, stride 2
            resnet.bn1,
            resnet.relu
        )                   # output: 64 x H/2 x W/2

        self.maxpool = resnet.maxpool   # output: 64 x H/4 x W/4

        self.enc1 = resnet.layer1      # output: 256  x H/4  x W/4
        self.enc2 = resnet.layer2      # output: 512  x H/8  x W/8
        self.enc3 = resnet.layer3      # output: 1024 x H/16 x W/16
        self.enc4 = resnet.layer4      # output: 2048 x H/32 x W/32  (bottleneck)

        # ---------------- Decoder ----------------
        # Up4: 2048 → 512, then cat with x3 (1024) → dec4
        self.up4  = nn.ConvTranspose2d(encoder_channels[4], 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(encoder_channels[3] + 512, 512)   # 1024 + 512 = 1536

        # Up3: 512 → 256, then cat with x2 (512) → dec3
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(encoder_channels[2] + 256, 256)   # 512 + 256 = 768

        # Up2: 256 → 128, then cat with x1 (256) → dec2
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(encoder_channels[1] + 128, 128)   # 256 + 128 = 384

        # Up1: 128 → 64, then cat with x0 (64) → dec1
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(encoder_channels[0] + 64, 64)     # 64 + 64 = 128

        # Final upsample to restore full resolution (H/2 → H)
        self.up_final = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # Output layer: 64 → 1 binary mask
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]   # save original H x W for safety check

        # ---------------- Encoder ----------------
        x0 = self.inconv(x)        # 64  x H/2  x W/2
        x1 = self.maxpool(x0)      # 64  x H/4  x W/4
        x1 = self.enc1(x1)         # 256 x H/4  x W/4

        x2 = self.enc2(x1)         # 512  x H/8  x W/8
        x3 = self.enc3(x2)         # 1024 x H/16 x W/16
        x4 = self.enc4(x3)         # 2048 x H/32 x W/32

        # ---------------- Decoder ----------------
        # Stage 4
        d4 = self.up4(x4)
        if d4.shape[2:] != x3.shape[2:]:
            d4 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, x3], dim=1)   # 1024 + 512 = 1536
        d4 = self.dec4(d4)                 # 512

        # Stage 3
        d3 = self.up3(d4)
        if d3.shape[2:] != x2.shape[2:]:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, x2], dim=1)   # 512 + 256 = 768
        d3 = self.dec3(d3)                 # 256

        # Stage 2
        d2 = self.up2(d3)
        if d2.shape[2:] != x1.shape[2:]:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x1], dim=1)   # 256 + 128 = 384
        d2 = self.dec2(d2)                 # 128

        # Stage 1
        d1 = self.up1(d2)
        if d1.shape[2:] != x0.shape[2:]:
            d1 = F.interpolate(d1, size=x0.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, x0], dim=1)   # 64 + 64 = 128
        d1 = self.dec1(d1)                 # 64

        # Final upsample: H/2 → H (restore full resolution)
        out = self.up_final(d1)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        out = self.final(out)              # 1 x H x W  (logits, no sigmoid)

        return out


# -----------------------
# Quick Sanity Check
# -----------------------
if __name__ == "__main__":
    model = UNetResNet(pretrained=False, num_classes=1)
    x = torch.randn(2, 3, 256, 256)   # batch=2, RGB, 256x256
    out = model(x)
    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == torch.Size([2, 1, 256, 256]), "Shape mismatch!"
    print("Sanity check passed!")