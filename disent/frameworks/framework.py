import torch


# ========================================================================= #
# framework                                                                 #
# ========================================================================= #


class BaseFramework(object):
    
    def training_step(self, model: torch.nn.Module, batch):
        raise NotImplementedError
    
    def validation_step(self, model: torch.nn.Module, batch):
        pass
    
    def test_step(self, model: torch.nn.Module, batch):
        pass


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
