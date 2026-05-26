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
# Lightweight GRA Block
# -----------------------
class GRABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, pred):
        # pred = coarse prediction logits
        # reverse attention map: background/confusing region focus
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
# SINet + GRA Model
# -----------------------
class SINet_GRA(nn.Module):
    def __init__(self, pretrained=True):
        super(SINet_GRA, self).__init__()

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
        self.search_up = nn.ConvTranspose2d(512, 256, 2, stride=2)

        # Coarse prediction from search stage
        self.coarse_pred = nn.Conv2d(256, 1, kernel_size=1)

        # -------- Identification Module --------
        self.id_conv1 = ConvBlock(256 + 1024, 256)
        self.gra1 = GRABlock(256)

        self.id_conv2 = ConvBlock(256 + 512, 128)
        self.gra2 = GRABlock(128)

        self.id_conv3 = ConvBlock(128 + 256, 64)
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
        s = self.search_up(s)

        # coarse prediction for reverse guidance
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
        d1 = self.gra1(d1, coarse)

        # prediction after stage 1
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
        d2 = self.gra2(d2, p1)

        # prediction after stage 2
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