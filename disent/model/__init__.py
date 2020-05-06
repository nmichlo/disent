
from .encoders_decoders import (
    EncoderSimpleFC,
    DecoderSimpleFC,
    EncoderSimpleConv64,
    DecoderSimpleConv64
)

from .gaussian_model import (
    GaussianEncoderDecoderModel
)

# ========================================================================= #
# __init__                                                                   #
# ========================================================================= #


def make_optimizer(name, parameters, lr=0.001):
    # SGD
    if 'sgd' == name:
        import torch.optim
        return torch.optim.SGD(parameters, lr=lr)
    # ADAM
    elif 'adam' == name:
        import torch.optim
        return torch.optim.Adam(parameters, lr=lr)
    # RADAM
    elif 'radam' == name:
        import torch_optimizer
        return torch_optimizer.RAdam(parameters, lr=lr)
    # UNKNOWN
    else:
        raise KeyError(f'Unsupported Optimizer: {name}')


def make_model(name, z_size=6, image_size=64, num_channels=3):
    from disent.model.gaussian_model import GaussianEncoderDecoderModel

    x_shape = (num_channels, image_size, image_size)

    if 'simple-fc' == name:
        from disent.model.encoders_decoders import EncoderSimpleFC, DecoderSimpleFC
        encoder = EncoderSimpleFC(x_shape=x_shape, h_size1=192, h_size2=128, z_size=z_size)  # 2.5 mil params... yoh
        decoder = DecoderSimpleFC(x_shape=x_shape, h_size1=192, h_size2=128, z_size=z_size)  # 2.5 mil params... yoh
    elif 'simple-conv' == name:
        from disent.model.encoders_decoders import EncoderSimpleConv64, DecoderSimpleConv64
        encoder = EncoderSimpleConv64(x_shape=x_shape, z_size=z_size)
        decoder = DecoderSimpleConv64(x_shape=x_shape, z_size=z_size)
    else:
        raise KeyError(f'Unsupported Model: {name}')

    return GaussianEncoderDecoderModel(encoder, decoder)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
