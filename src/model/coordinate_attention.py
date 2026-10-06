import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: int = 32,
    ):
        super().__init__()

        reduced_channels = max(
            8,
            channels // reduction,
        )

        self.conv1 = nn.Conv2d(
            channels,
            reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            reduced_channels
        )

        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(
            reduced_channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.conv_w = nn.Conv2d(
            reduced_channels,
            channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):

        identity = x

        h, w = x.shape[2:]

        pool_h = x.mean(
            dim=3,
            keepdim=True,
        )

        pool_w = x.mean(
            dim=2,
            keepdim=True,
        )

        pool_w = pool_w.permute(
            0, 1, 3, 2
        )

        y = torch.cat(
            [pool_h, pool_w],
            dim=2,
        )

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y_h, y_w = torch.split(
            y,
            [h, w],
            dim=2,
        )

        y_w = y_w.permute(
            0, 1, 3, 2
        )

        attention_h = torch.sigmoid(
            self.conv_h(y_h)
        )

        attention_w = torch.sigmoid(
            self.conv_w(y_w)
        )

        return (
            identity
            * attention_h
            * attention_w
        )