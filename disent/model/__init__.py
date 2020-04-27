
from .encoders_decoders import (
    EncoderSimpleFC,
    DecoderSimpleFC,
    EncoderSimpleConv64,
    DecoderSimpleConv64
)

from .gaussian_encoder_model import (
    GaussianEncoderModel
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


def make_model(name, z_dim=6, image_size=64, num_channels=3):
    from disent.model.gaussian_encoder_model import GaussianEncoderModel

    if 'simple-fc' == name:
        from disent.model.encoders_decoders import EncoderSimpleFC, DecoderSimpleFC
        encoder = EncoderSimpleFC(x_dim=(image_size**2)*num_channels, h_dim1=256, h_dim2=128, z_dim=z_dim)  # 3 mil params... yoh
        decoder = DecoderSimpleFC(x_dim=(image_size**2)*num_channels, h_dim1=256, h_dim2=128, z_dim=z_dim)  # 3 mil params... yoh
    elif 'simple-conv' == name:
        from disent.model.encoders_decoders import EncoderSimpleConv64, DecoderSimpleConv64
        encoder = EncoderSimpleConv64(latent_dim=z_dim, num_channels=3, image_size=64)
        decoder = DecoderSimpleConv64(latent_dim=z_dim, num_channels=3, image_size=64)
    else:
        raise KeyError(f'Unsupported Model: {name}')

    return GaussianEncoderModel(encoder, decoder)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
