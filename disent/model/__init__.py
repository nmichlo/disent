
from disent.model.encoders_decoders import (
    EncoderSimpleFC,
    DecoderSimpleFC,
    EncoderSimpleConv64,
    DecoderSimpleConv64,
    EncoderFC,
    DecoderFC,
    EncoderConv64,
    DecoderConv64,
)

from disent.model.gaussian_model import (
    GaussianEncoderDecoderModel
)

# ========================================================================= #
# __init__                                                                   #
# ========================================================================= #


def make_optimizer(name, parameters, lr=0.001, weight_decay=.0):
    # SGD
    if 'sgd' == name:
        import torch.optim
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    # ADAM
    elif 'adam' == name:
        import torch.optim
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    # RADAM
    elif 'radam' == name:
        import torch_optimizer
        return torch_optimizer.RAdam(parameters, lr=lr, weight_decay=weight_decay)
    # UNKNOWN
    else:
        raise KeyError(f'Unsupported Optimizer: {name}')


def make_model(name, z_size=6, image_size=64, num_channels=3):
    from disent.model.gaussian_model import GaussianEncoderDecoderModel

    x_shape = (num_channels, image_size, image_size)

    if 'simple-fc' == name:
        encoder = EncoderSimpleFC(x_shape=x_shape, h_size1=192, h_size2=128, z_size=z_size)  # 2.5 mil params... yoh
        decoder = DecoderSimpleFC(x_shape=x_shape, h_size1=192, h_size2=128, z_size=z_size)  # 2.5 mil params... yoh
    elif 'simple-conv' == name:
        encoder = EncoderSimpleConv64(x_shape=x_shape, z_size=z_size)
        decoder = DecoderSimpleConv64(x_shape=x_shape, z_size=z_size)
    elif 'fc' == name:
        encoder = EncoderFC(x_shape=x_shape, z_size=z_size)
        decoder = DecoderFC(x_shape=x_shape, z_size=z_size)
    elif 'conv' == name:
        encoder = EncoderConv64(x_shape=x_shape, z_size=z_size, dropout=0.0)
        decoder = DecoderConv64(x_shape=x_shape, z_size=z_size, dropout=0.0)
    elif 'conv-dropout' == name:
        encoder = EncoderConv64(x_shape=x_shape, z_size=z_size, dropout=0.33)
        decoder = DecoderConv64(x_shape=x_shape, z_size=z_size, dropout=0.33)
    else:
        raise KeyError(f'Unsupported Model: {name}')

    return GaussianEncoderDecoderModel(encoder, decoder)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    import torch

    for num_channels in [1, 3]:
        for z_size in [3, 12]:
            for name in ['simple-conv', 'simple-fc', 'conv', 'fc']:
                try:
                    print(f'NAME: {name:11s} Z: {z_size:2d} CHANNELS: {num_channels}', end=' | ', flush=True)
                    model = make_model(name, z_size=z_size, num_channels=num_channels)
                    batch = torch.randn(16, num_channels, 64, 64)
                    out = model(batch)
                    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)
                except Exception as e:
                    print(f'\033[91mFAILED: {e}\033[0m')

