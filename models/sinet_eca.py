import torch
import torch.nn.functional as F
import torch.nn as nn


# -----------------------
# Basic Conv Block
# -----------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# -----------------------
# ECA Attention Block
# -----------------------
class ECABlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]

        y = self.avg_pool(x)          # [B, C, 1, 1]
        y = y.squeeze(-1)             # [B, C, 1]
        y = y.transpose(-1, -2)       # [B, 1, C]

        y = self.conv(y)              # [B, 1, C]
        y = self.sigmoid(y)

        y = y.transpose(-1, -2)       # [B, C, 1]
        y = y.unsqueeze(-1)           # [B, C, 1, 1]

        return x * y.expand_as(x)


# -----------------------
# SINet + ECA Model
# -----------------------
class SINet_ECA(nn.Module):
    def __init__(self, pretrained=True):
        super(SINet_ECA, self).__init__()

        # -------- Encoder (ResNet50) --------
        from torchvision.models import resnet50, ResNet50_Weights

        if pretrained:
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            resnet = resnet50(weights=None)

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.pool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256
        self.layer2 = resnet.layer2  # 512
        self.layer3 = resnet.layer3  # 1024
        self.layer4 = resnet.layer4  # 2048

        # -------- Search Module --------
        self.search_conv = ConvBlock(2048, 512)
        self.eca_search = ECABlock(512)

        self.search_up = nn.ConvTranspose2d(512, 256, 2, stride=2)

        # -------- Identification Module --------
        self.id_conv1 = ConvBlock(256 + 1024, 256)
        self.eca_id1 = ECABlock(256)

        self.id_conv2 = ConvBlock(256 + 512, 128)
        self.eca_id2 = ECABlock(128)

        self.id_conv3 = ConvBlock(128 + 256, 64)
        self.eca_id3 = ECABlock(64)

        # -------- Final Output --------
        self.final_up = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        x0 = self.conv1(x)
        x1 = self.pool(x0)

        x2 = self.layer1(x1)   # 256
        x3 = self.layer2(x2)   # 512
        x4 = self.layer3(x3)   # 1024
        x5 = self.layer4(x4)   # 2048

        # -------- Search Stage --------
        s = self.search_conv(x5)
        s = self.eca_search(s)     # ECA added here
        s = self.search_up(s)

        # -------- Stage 1 --------
        s = F.interpolate(
            s, size=x4.shape[2:], mode='bilinear', align_corners=False
        )

        d1 = torch.cat([s, x4], dim=1)
        d1 = self.id_conv1(d1)
        d1 = self.eca_id1(d1)      # ECA added here

        # -------- Stage 2 --------
        d1 = F.interpolate(
            d1, size=x3.shape[2:], mode='bilinear', align_corners=False
        )

        d2 = torch.cat([d1, x3], dim=1)
        d2 = self.id_conv2(d2)
        d2 = self.eca_id2(d2)      # ECA added here

        # -------- Stage 3 --------
        d2 = F.interpolate(
            d2, size=x2.shape[2:], mode='bilinear', align_corners=False
        )

        d3 = torch.cat([d2, x2], dim=1)
        d3 = self.id_conv3(d3)
        d3 = self.eca_id3(d3)      # ECA added here

        # -------- Output --------
        out = self.final_up(d3)
        out = self.out(out)

        out = F.interpolate(
            out,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return out