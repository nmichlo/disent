

# ========================================================================= #
# make - optimizers                                                         #
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


# ========================================================================= #
# make - models                                                             #
# ========================================================================= #


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
# make - datasets                                                           #
# ========================================================================= #


def make_ground_truth_dataset(name, data_dir='data', try_in_memory=True):
    import torchvision
    import torch

    # all datasets are in the range [0, 1] to correspond
    # to the sigmoid activation function.

    if '3dshapes' == name:
        from disent.dataset.ground_truth.dataset_shapes3d import Shapes3dDataset
        return Shapes3dDataset(data_dir=data_dir, transform=torchvision.transforms.ToTensor())

    elif 'dsprites' == name:
        from disent.dataset.ground_truth.dataset_dsprites import DSpritesDataset, DSpritesMemoryDataset
        return (DSpritesMemoryDataset if try_in_memory else DSpritesDataset)(data_dir=data_dir, transform=torchvision.transforms.ToTensor())

    elif 'xygrid' == name:
        from disent.dataset.ground_truth.dataset_xygrid import XYDataset
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0))  # add extra channels
        ])
        return XYDataset(width=64, transform=transforms)

    elif 'mnist' == name:
        # this dataset is quite pointless, just compatible with the others...
        from PIL import Image
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(64, interpolation=Image.NEAREST),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),  # add extra channels
        ])
        # MNIST that only returns images, without labels
        # torchvision.datasets.MNIST names data folder based off of class name
        class MNIST(torchvision.datasets.MNIST):
            def __getitem__(self, index):
                return super().__getitem__(index)[0]
        # return instance
        return MNIST(root=data_dir, train=True, transform=transforms, download=True)

    else:
        raise KeyError(f'Unsupported Ground Truth Dataset: {name}')

# def check_values(name):
#     item = make_ground_truth_dataset(name)[0]
#     print(name, item.min(), item.max())
# [check(name) for name in ['3dshapes', 'xygrid', 'mnist']]

def make_paired_dataset(name, k='uniform'):
    from disent.dataset.ground_truth.base import PairedVariationDataset
    dataset = make_ground_truth_dataset(name)
    return PairedVariationDataset(dataset, k=k)


# ========================================================================= #
# make - loss                                                               #
# ========================================================================= #


def make_vae_loss(name):
    if 'vae' == name:
        from disent.loss.loss import VaeLoss
        return VaeLoss()
    elif 'beta-vae' == name:
        from disent.loss.loss import BetaVaeLoss
        return BetaVaeLoss(beta=4)
    elif 'beta-vae-h' == name:
        raise NotImplementedError('beta-vae-h loss is not yet implemented')
    elif 'ada-gvae' == name:
        from disent.loss.loss import AdaGVaeLoss
        return AdaGVaeLoss(beta=4)
    elif 'ada-ml-vae' == name:
        from disent.loss.loss import AdaMlVaeLoss
        return AdaMlVaeLoss(beta=4)
    else:
        raise KeyError(f'Unsupported Ground Truth Dataset: {name}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
