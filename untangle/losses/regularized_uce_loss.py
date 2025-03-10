"""Regularized UCE loss for PostNets."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Dirichlet


class RegularizedUCELoss(nn.Module):
    """Implements the Regularized Uncertain Cross-Entropy (UCE) loss for PostNets.

    This loss combines uncertain cross-entropy with Dirichlet entropy regularization.

    Args:
        regularization_factor: Weight for the entropy regularization term.
    """

    def __init__(self, regularization_factor: float) -> None:
        super().__init__()

        self.regularization_factor = regularization_factor

    def forward(self, alphas: Tensor, targets: Tensor) -> Tensor:
        """Compute the Regularized UCE loss.

        Args:
            alphas: The alpha parameters of the Dirichlet distribution.
            targets: The target class labels.

        Returns:
            The computed Regularized UCE loss.
        """
        targets_one_hot = F.one_hot(targets, num_classes=alphas.shape[1])  # [B, C]

        sum_alphas = alphas.sum(dim=1)  # [B]
        entropy_regularizer = Dirichlet(alphas).entropy()  # [B]

        return (
            torch.sum(
                targets_one_hot
                * (torch.digamma(sum_alphas.unsqueeze(1)) - torch.digamma(alphas)),
                dim=1,
            ).mean()
            - self.regularization_factor * entropy_regularizer.mean()
        )
