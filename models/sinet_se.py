import torch
import torch.nn as nn
import torch.nn.functional as F


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
# SE Attention Block
# -----------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),

            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


# -----------------------
# SINet + SE
# -----------------------
class SINet_SE(nn.Module):
    def __init__(self, pretrained=True):
        super(SINet_SE, self).__init__()

        # -----------------------
        # ResNet50 Encoder
        # -----------------------
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

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # -----------------------
        # Search Module
        # -----------------------
        self.search_conv = ConvBlock(2048, 512)

        self.se_search = SEBlock(512)

        self.search_up = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        # -----------------------
        # Identification Module
        # -----------------------
        self.id_conv1 = ConvBlock(256 + 1024, 256)
        self.se1 = SEBlock(256)

        self.id_conv2 = ConvBlock(256 + 512, 128)
        self.se2 = SEBlock(128)

        self.id_conv3 = ConvBlock(128 + 256, 64)
        self.se3 = SEBlock(64)

        # -----------------------
        # Output
        # -----------------------
        self.final_up = nn.ConvTranspose2d(
            64,
            64,
            kernel_size=2,
            stride=2
        )

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        # -----------------------
        # Encoder
        # -----------------------
        x0 = self.conv1(x)
        x1 = self.pool(x0)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # -----------------------
        # Search Stage
        # -----------------------
        s = self.search_conv(x5)

        # SE added
        s = self.se_search(s)

        s = self.search_up(s)

        # -----------------------
        # Stage 1
        # -----------------------
        s = F.interpolate(
            s,
            size=x4.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        d1 = torch.cat([s, x4], dim=1)

        d1 = self.id_conv1(d1)

        # SE added
        d1 = self.se1(d1)

        # -----------------------
        # Stage 2
        # -----------------------
        d1 = F.interpolate(
            d1,
            size=x3.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        d2 = torch.cat([d1, x3], dim=1)

        d2 = self.id_conv2(d2)

        # SE added
        d2 = self.se2(d2)

        # -----------------------
        # Stage 3
        # -----------------------
        d2 = F.interpolate(
            d2,
            size=x2.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        d3 = torch.cat([d2, x2], dim=1)

        d3 = self.id_conv3(d3)

        # SE added
        d3 = self.se3(d3)

        # -----------------------
        # Output
        # -----------------------
        out = self.final_up(d3)

        out = self.out(out)

        out = F.interpolate(
            out,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return out