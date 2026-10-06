import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttentionGate(nn.Module):

    def __init__(
        self,
        x_channels: int,
        g_channels: int,
        inter_channels: int,
    ):
        super().__init__()

        # omega_x in the paper
        self.theta_x = nn.Conv2d(
            x_channels,
            inter_channels,
            kernel_size=1,
            bias=True,
        )

        # omega_g in the paper
        self.phi_g = nn.Conv2d(
            g_channels,
            inter_channels,
            kernel_size=1,
            bias=True,
        )

        # psi in the paper
        self.psi = nn.Conv2d(
            inter_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        # sigma_1
        self.relu = nn.ReLU(inplace=True)

        # sigma_2 for attention coefficient
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):

        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)

        # Match spatial dimensions when necessary.
        if theta_x.shape[2:] != phi_g.shape[2:]:
            phi_g = F.interpolate(
                phi_g,
                size=theta_x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # alpha =
        # sigma_2(
        #     psi(
        #         sigma_1(
        #             omega_x * x_l +
        #             omega_g * g_l +
        #             b_g
        #         )
        #     )
        #     + b_psi
        # )
        attention = self.relu(
            theta_x + phi_g
        )

        attention = self.psi(
            attention
        )

        attention = self.sigmoid(
            attention
        )

        # x_hat_l = alpha * x_l
        filtered = x * attention

        # g_l and x_hat_l must have matching spatial dimensions.
        if filtered.shape[2:] != g.shape[2:]:
            filtered = F.interpolate(
                filtered,
                size=g.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # x_hat_out = sigma_2(x_hat_l + g_l)
        return self.relu(
            filtered + g
        )