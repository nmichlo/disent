import pytorch_lightning as pl
from pytorch_lightning import Trainer
from dataset import XYDataset
from model import XYNet
import torch.nn.functional as F
import torch_optimizer as optim_extra
import torch
import torch.utils.data
import numpy as np

# ========================================================================= #
# xy system                                                                 #
# ========================================================================= #


class XYSystem(pl.LightningModule):
    def __init__(self, size=8, arch='decoder', batch_size=16):
        super().__init__()
        self.size = size
        self.arch = arch
        self.validation_count = 0
        self.batch_size = batch_size
        self.model = XYNet(size=size, arch=arch)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        if self.arch == 'encoder':
            loss = F.mse_loss(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y)
        return {
            'loss': loss,
            'log': {'train_loss': loss}
        }

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=0.001, weight_decay=0.00001)
        return optim_extra.RAdam(self.parameters(), lr=0.001, weight_decay=0.00001)

    def validation_step(self, batch, batch_idx):
        return {
            'val_loss': self.training_step(batch, batch_idx)['loss'],
            # 'val_in': batch,
            # 'val_out': self.forward(batch),
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Print Extra Info
        if self.validation_count % 50 == 0:
            print(f'[{self.validation_count}] Ended Validation: {avg_loss}')
        self.validation_count += 1

        return {
            'avg_val_loss': avg_loss,
            'log': {'val_loss': avg_loss}
        }

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(XYDataset(size=self.size, arch=self.arch), batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(XYDataset(size=self.size, arch=self.arch), batch_size=self.batch_size)


def compare_system(system, size):
    encodings = []
    for idx in range(size ** 2):
        encoded, decoded = XYDataset.gen_pair(size=size, idx=idx)

        encoding = system.model.encode(torch.as_tensor(decoded))[0].detach().numpy() if (system.arch != 'decoder') else encoded
        decoding = system.model.decode(torch.as_tensor(encoding))[0].detach().numpy() if (system.arch != 'encoder') else decoded

        # decoding = model.forward(torch.as_tensor(decoded))[0].detach().numpy()

        # out_pos = reverse_pos(GRID_SIZE, encoding.detach().numpy())
        encodings.append({
            "encoded": encoded,
            "decoded": decoded,
            "idx": idx,
            # "pos": idx2pos(GRID_SIZE, idx),
            "out_encoding": encoding,
            "out_decoding": decoding,
            "out_idx": np.argmax(decoding),
            # "out_pos_raw": out_pos,
            # "out_pos": np.floor(out_pos),
            # "out_idx": pos2idx(GRID_SIZE, np.floor(out_pos))

        })
    return encodings


# ========================================================================= #
# Main                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    GRID_SIZE = 8

    # Initialise System
    system = XYSystem(size=GRID_SIZE, arch='decoder') # arch is one of: encoder, decoder, full

    # Train System
    trainer = Trainer(max_epochs=2000, show_progress_bar=False)
    trainer.fit(system)

    # Compare System
    dats = compare_system(system, GRID_SIZE)
    for dat in dats:
        print(dat['decoded'])
        print(np.reshape(np.around(dat['out_decoding'], 3), [GRID_SIZE, GRID_SIZE]))
        print()
    print([dat["idx"] for dat in dats])
    print([dat["out_idx"] for dat in dats])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
