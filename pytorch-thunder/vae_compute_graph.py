
import pythonflow as pf
import torch
import pytorch_lightning as pl
import torchvision
from torch_optimizer import RAdam
from tqdm import tqdm

from disent.dataset import GroundTruthDataset
from disent.dataset.ground_truth_data.data_xygrid import XYData
from disent.frameworks.unsupervised.vae import bce_loss_with_logits, kl_normal_loss
from disent.model import DecoderConv64, EncoderConv64

# ========================================================================= #
# graph_vae                                                                 #
# ========================================================================= #


bce_loss_with_logits = pf.opmethod(bce_loss_with_logits)

kl_normal_loss = pf.opmethod(kl_normal_loss)


@pf.opmethod(length=2)
def split_in_two(x):
    num = x.shape[1]
    return x[:, :num // 2], x[:, num // 2:]


@pf.opmethod()
def reparameterize(z_mean, z_logvar):
    std = torch.exp(0.5 * z_logvar)  # std == var^0.5 == e^(log(var^0.5)) == e^(0.5*log(var))
    eps = torch.randn_like(std)  # N(0, 1)
    return z_mean + (std * eps)  # mu + dot(std, eps)


@pf.opmethod()
def make_dict(**kwargs):
    return kwargs


# ========================================================================= #
# VAE                                                                       #
# ========================================================================= #


with pf.Graph() as vae_graph:
    # INPUTS
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    encoder = pf.placeholder(name='encoder')
    decoder = pf.placeholder(name='decoder')
    x = pf.placeholder(name='x')
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    
    # FORWARD - TRAIN
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # latent distribution parametrisation
    z_mean, z_logvar = split_in_two(encoder(x))
    z_mean.name, z_logvar.name = 'z_mean', 'z_logvar'
    # sample from latent distribution
    z = reparameterize(z_mean, z_logvar).set_name('z')
    # reconstruct (without the final activation)
    x_recon = decoder(z).set_name('x_recon')
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    
    # FORWARD
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    forward = decoder(z_mean).set_name('forward')
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    
    # LOSS
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # reconstruction error
    recon_loss = bce_loss_with_logits(x, x_recon).set_name('recon_loss')
    # KL divergence
    kl_loss = kl_normal_loss(z_mean, z_logvar).set_name('kl_loss')
    # compute combined loss
    loss = (recon_loss + kl_loss).set_name('loss')
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    
    # LOG
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    log = make_dict(
        loss=loss,
        recon_loss=recon_loss,
        kl_loss=kl_loss,
        elbo=-(recon_loss + kl_loss)
    ).set_name('log')
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #


# ========================================================================= #
# MODULE                                                                    #
# ========================================================================= #


class GraphLightningModule(pl.LightningModule):
    
    def __init__(self, graph, **graph_kwargs):
        super().__init__()
        self.compute_graph = graph
        self.graph_kwargs = torch.nn.ModuleDict(graph_kwargs)
    
    def forward(self, batch):
        return self.compute_graph('forward', {
            'x': batch,
            **self.graph_kwargs
        })
    
    def training_step(self, batch, batch_idx):
        loss, log = self.compute_graph(['loss', 'log'], {
            'x': batch,
            **self.graph_kwargs
        })
        return {
            'loss': loss,
            'log': log,
        }
    
    def configure_optimizers(self):
        return RAdam(self.parameters())


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    
    data = XYData()
    dataset = GroundTruthDataset(data, transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=6, shuffle=True)
    
    
    model = GraphLightningModule(
        vae_graph,
        encoder=EncoderConv64(z_multiplier=2),
        decoder=DecoderConv64(),
    ).cuda()
    
    # About the same speed
    # model = Vae(
    #     make_optimiser_fn=RAdam,
    #     make_model_fn=lambda: GaussianAutoEncoder(EncoderConv64(z_multiplier=2), DecoderConv64())
    # )
    
    trainer = pl.Trainer(max_epochs=10, gpus=1)
    trainer.fit(model, train_dataloader=dataloader)

    model = model.cuda() # trainer resets this

    # 470 iter/s:
    for batch in tqdm(dataloader):
        model.forward(batch.cuda())

    # 500 iter/s:
    with torch.no_grad():
        for batch in tqdm(dataloader):
            model.forward(batch.cuda())


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
