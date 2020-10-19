import logging
from disent.util import DisentModule


log = logging.getLogger(__name__)


# ========================================================================= #
# Utility Layers                                                            #
# ========================================================================= #


class Print(DisentModule):
    """From: https://github.com/1Konny/Beta-VAE/blob/master/model.py"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, tensor):
        log.debug(self.layer, '|', tensor.shape, '->')
        output = self.layer.forward(tensor)
        log.debug(output.shape)
        return output


class BatchView(DisentModule):
    """From: https://github.com/1Konny/Beta-VAE/blob/master/model.py"""
    def __init__(self, size):
        super().__init__()
        self.size = (-1, *size)

    def forward(self, tensor):
        return tensor.view(*self.size)


class Unsqueeze3D(DisentModule):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


class Flatten3D(DisentModule):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
