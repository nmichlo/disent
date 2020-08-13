import torch

from disent.frameworks.framework import BaseFramework


# ========================================================================= #
# simclr                                                                   #
# ========================================================================= #


class SimCLR(BaseFramework):
    # https://github.com/Spijkervet/SimCLR
    # https://github.com/leftthomas/SimCLR
    # https://github.com/sthalles/SimCLR
    # https://github.com/sthalles/PyTorch-BYOL
    
    def __init__(self, make_optimizer_fn):
        super().__init__(make_optimizer_fn)

    def forward(self, batch) -> torch.Tensor:
        pass

    def compute_loss(self, batch, batch_idx) -> dict:
        pass


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
