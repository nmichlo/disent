import torch


# ========================================================================= #
# framework                                                                 #
# ========================================================================= #


class BaseFramework(object):
    
    def training_step(self, model: torch.nn.Module, batch) -> dict:
        raise NotImplementedError
    
    def validation_step(self, model: torch.nn.Module, batch) -> dict:
        raise NotImplementedError

    def test_step(self, model: torch.nn.Module, batch) -> dict:
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
