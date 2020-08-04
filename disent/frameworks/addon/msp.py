import logging

import torch
import numpy as np
from torch.nn import functional as F
from disent.model.base import BaseModule

log = logging.getLogger(__name__)


class MatrixSubspaceProjection(BaseModule):
    """
    Matrix Subspace Projection Module
    - Name: Latent Space Factorisation and Manipulation via Matrix Subspace Projection
    - Paper: https://arxiv.org/abs/1907.12385
    - Home Page: https://xiao.ac/proj/msp
    - Original Code: https://github.com/lissomx/MSP
    """

    def __init__(self, y_size, x_shape=(64, 64, 3), z_size=6, z_multiplier=1, init_mode='xavier'):
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)
        # U = [M, N],      N is the null space of M, N perpendicular to M
        # U is orthogonal, ie ... U^T ≡ U^−1
        # M -> y | attribute information
        self.y_size = y_size
        # learnable weights:
        self.M = torch.nn.Parameter(torch.empty(self.y_size, self.z_total), requires_grad=True)
        # Initialise M
        if init_mode == 'xavier':
            torch.nn.init.xavier_normal_(self.M)
        elif init_mode == 'ortho':
            # in this mode, if all labels are zero and passed to mutated_z, the result should be all zeros.
            # typically the loss enforces this constraint that M^T = M^-1
            from scipy.stats import ortho_group
            assert self.y_size == self.z_size, f'For {init_mode=}, {y_size=} must equal {z_size=}'
            self.M[:, :] = torch.as_tensor(ortho_group.rvs(dim=self.y_size))
        else:
            raise KeyError(f'Invalid {init_mode=}')
        
    @property
    def msp_loss_weight(self):
        return np.prod(self.x_size) / (self.z_total + self.y_size)

    def msp_loss(self, z, label):
        # L1 = F.mse_loss((z @ self.M.T).view(-1), label.view(-1), reduction="none").sum()
        # L2 = F.mse_loss((label @ self.M).view(-1), z.view(-1), reduction="none").sum()
        L1 = F.mse_loss((z @ self.M.T), label, reduction="none").sum()
        L2 = F.mse_loss((label @ self.M), z, reduction="none").sum()
        return L1 + L2

    def loss(self, z, label):
        return self.msp_loss_weight * self.msp_loss(z, label)

    def loss_batch(self, z, label):
        assert z.ndim == 2
        return self.msp_loss(z, label) / ((self.z_total + self.y_size) * len(z))

    def mutated_z(self, z, label_vec, apply_label_mask):
        """
        label_vec is the desired label values
        apply_label_mask is the boolean mask of which label changes to apply
        """
        y_hat = z @ self.M.T
        # compute delta in label space
        delta = (label_vec - y_hat) * apply_label_mask
        # apply delta in latent space
        return z + delta @ self.M

    def latent_to_labels(self, z):
        return z @ self.M.T

    def labels_to_latent(self, label):
        return label @ self.M
