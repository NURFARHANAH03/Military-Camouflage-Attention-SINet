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
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * y.expand_as(x)


# -----------------------
# Lightweight GRA Block
# -----------------------
class GRABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, pred):
        reverse_map = 1 - torch.sigmoid(pred)

        reverse_map = F.interpolate(
            reverse_map,
            size=feat.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        guided_feat = torch.cat([feat, reverse_map], dim=1)
        refined = self.refine(guided_feat)

        return feat + refined


# -----------------------
# SINet + ECA + GRA Model
# -----------------------
class SINet_ECA_GRA(nn.Module):
    def __init__(self, pretrained=True):
        super(SINet_ECA_GRA, self).__init__()

        from torchvision.models import resnet50, ResNet50_Weights

        if pretrained:
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            resnet = resnet50(weights=None)

        # -------- Encoder --------
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

        # coarse prediction for GRA
        self.coarse_pred = nn.Conv2d(256, 1, kernel_size=1)

        # -------- Identification Module --------
        self.id_conv1 = ConvBlock(256 + 1024, 256)
        self.eca_id1 = ECABlock(256)
        self.gra1 = GRABlock(256)

        self.id_conv2 = ConvBlock(256 + 512, 128)
        self.eca_id2 = ECABlock(128)
        self.gra2 = GRABlock(128)

        self.id_conv3 = ConvBlock(128 + 256, 64)
        self.eca_id3 = ECABlock(64)
        self.gra3 = GRABlock(64)

        # -------- Final Output --------
        self.final_up = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        x0 = self.conv1(x)
        x1 = self.pool(x0)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # -------- Search Stage --------
        s = self.search_conv(x5)
        s = self.eca_search(s)
        s = self.search_up(s)

        coarse = self.coarse_pred(s)

        # -------- Stage 1 --------
        s = F.interpolate(
            s,
            size=x4.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d1 = torch.cat([s, x4], dim=1)
        d1 = self.id_conv1(d1)
        d1 = self.eca_id1(d1)
        d1 = self.gra1(d1, coarse)

        p1 = self.out_intermediate(d1, x.shape[2:])

        # -------- Stage 2 --------
        d1 = F.interpolate(
            d1,
            size=x3.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d2 = torch.cat([d1, x3], dim=1)
        d2 = self.id_conv2(d2)
        d2 = self.eca_id2(d2)
        d2 = self.gra2(d2, p1)

        p2 = self.out_intermediate(d2, x.shape[2:])

        # -------- Stage 3 --------
        d2 = F.interpolate(
            d2,
            size=x2.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d3 = torch.cat([d2, x2], dim=1)
        d3 = self.id_conv3(d3)
        d3 = self.eca_id3(d3)
        d3 = self.gra3(d3, p2)

        # -------- Output --------
        out = self.final_up(d3)
        out = self.out(out)

        out = F.interpolate(
            out,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        return out

    def out_intermediate(self, feat, output_size):
        pred = feat.mean(dim=1, keepdim=True)

        pred = F.interpolate(
            pred,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        return pred