from torch import nn
from disent.util import colors as c
import logging

log = logging.getLogger(__name__)


def init_model_weights(model: nn.Module, mode='xavier_normal'):
    count = 0

    # get default mode
    if mode is None:
        mode = 'default'

    def init_normal(m):
        nonlocal count
        init, count = False, count + 1

        # actually initialise!
        if mode == 'xavier_normal':
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                init = True
        elif mode == 'default':
            pass
        else:
            raise KeyError(f'Unknown init mode: {repr(mode)}')

        # print messages
        if init:
            log.info(f'| {count:03d} {c.lGRN}INIT{c.RST}: {m.__class__.__name__}')
        else:
            log.info(f'| {count:03d} {c.lRED}SKIP{c.RST}: {m.__class__.__name__}')

    log.info(f'Initialising Model Layers: {mode}')
    model.apply(init_normal)

    return model