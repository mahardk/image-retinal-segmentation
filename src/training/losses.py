import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(
            probabilities.size(0),
            -1,
        )

        targets = targets.reshape(
            targets.size(0),
            -1,
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        return (
            self.bce(logits, targets)
            + self.dice(logits, targets)
        )