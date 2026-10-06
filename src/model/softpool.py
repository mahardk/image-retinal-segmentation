import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPool2d(nn.Module):

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):

        b, c, h, w = x.shape

        k = self.kernel_size

        patches = F.unfold(
            x,
            kernel_size=k,
            stride=self.stride,
        )

        patches = patches.view(
            b,
            c,
            k * k,
            -1,
        )

        weights = torch.softmax(
            patches,
            dim=2,
        )

        pooled = (
            patches * weights
        ).sum(dim=2)

        out_h = (
            (h - k) // self.stride
        ) + 1

        out_w = (
            (w - k) // self.stride
        ) + 1

        return pooled.view(
            b,
            c,
            out_h,
            out_w,
        )