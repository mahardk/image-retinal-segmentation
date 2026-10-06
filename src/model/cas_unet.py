import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.coordinate_attention import CoordinateAttention
from src.model.softpool import SoftPool2d
from src.model.attention_gate import AdditiveAttentionGate


class DCConvBlock(nn.Module):
    """
    DC-Conv Block used in the encoder/decoder.

    The paper describes the block using:
        Conv -> BN -> ReLU

    Coordinate Attention is inserted into the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_reduction: int = 32,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(inplace=True)

        self.ca = CoordinateAttention(
            out_channels,
            reduction=attention_reduction,
        )

    def forward(self, x):

        x = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = self.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = self.ca(x)

        return x


class CASUNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels=(32, 64, 128, 256, 512),
        attention_reduction: int = 32,
    ):
        super().__init__()

        c1, c2, c3, c4, c5 = channels

        # Encoder
        self.enc1 = DCConvBlock(
            in_channels,
            c1,
            attention_reduction,
        )

        self.enc2 = DCConvBlock(
            c1,
            c2,
            attention_reduction,
        )

        self.enc3 = DCConvBlock(
            c2,
            c3,
            attention_reduction,
        )

        self.enc4 = DCConvBlock(
            c3,
            c4,
            attention_reduction,
        )

        self.pool1 = SoftPool2d()
        self.pool2 = SoftPool2d()
        self.pool3 = SoftPool2d()
        self.pool4 = SoftPool2d()

        # Bottleneck
        self.bottleneck = DCConvBlock(
            c4,
            c5,
            attention_reduction,
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            c5,
            c4,
            kernel_size=2,
            stride=2,
        )

        self.ag4 = AdditiveAttentionGate(
            x_channels=c4,
            g_channels=c4,
            inter_channels=c4 // 2,
        )

        self.dec4 = DCConvBlock(
            c4,
            c4,
            attention_reduction,
        )

        self.up3 = nn.ConvTranspose2d(
            c4,
            c3,
            kernel_size=2,
            stride=2,
        )

        self.ag3 = AdditiveAttentionGate(
            x_channels=c3,
            g_channels=c3,
            inter_channels=c3 // 2,
        )

        self.dec3 = DCConvBlock(
            c3,
            c3,
            attention_reduction,
        )

        self.up2 = nn.ConvTranspose2d(
            c3,
            c2,
            kernel_size=2,
            stride=2,
        )

        self.ag2 = AdditiveAttentionGate(
            x_channels=c2,
            g_channels=c2,
            inter_channels=c2 // 2,
        )

        self.dec2 = DCConvBlock(
            c2,
            c2,
            attention_reduction,
        )

        self.up1 = nn.ConvTranspose2d(
            c2,
            c1,
            kernel_size=2,
            stride=2,
        )

        self.ag1 = AdditiveAttentionGate(
            x_channels=c1,
            g_channels=c1,
            inter_channels=c1 // 2,
        )

        self.dec1 = DCConvBlock(
            c1,
            c1,
            attention_reduction,
        )

        self.final_conv = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = self.ag4(e4, d4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.ag3(e3, d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.ag2(e2, d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.ag1(e1, d1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)