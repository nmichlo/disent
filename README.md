
<p align="center">
    <h1 align="center">🧶 Disent</h1>
    <p align="center">⚠️ W.I.P</p>
    <p align="center">
        <i>A modular disentangled representation learning framework for pytorch</i>
    </p>
</p>

<p align="center">
    <a href="https://choosealicense.com/licenses/mit/">
        <img alt="license" src="https://img.shields.io/github/license/nmichlo/disent?style=flat-square&color=lightgrey"/>
    </a>
    <a href="https://pypi.org/project/disent">
        <img alt="python versions" src="https://img.shields.io/pypi/pyversions/disent?style=flat-square"/>
    </a>
    <a href="https://pypi.org/project/disent">
        <img alt="pypi version" src="https://img.shields.io/pypi/v/disent?style=flat-square&color=blue"/>
    </a>
    <a href="https://github.com/nmichlo/disent/actions?query=workflow%3Atest">
        <img alt="tests status" src="https://img.shields.io/github/workflow/status/nmichlo/disent/test?label=tests&style=flat-square"/>
    </a>
<!--     <a href="https://codecov.io/gh/nmichlo/disent/"> -->
<!--         <img alt="code coverage" src="https://img.shields.io/codecov/c/gh/nmichlo/disent?token=86IZK3J038&style=flat-square"/> -->
<!--     </a> -->
<!--     <a href="https://github.com/nmichlo/disent"> -->
<!--         <img alt="last commit" src="https://img.shields.io/github/last-commit/nmichlo/disent?style=flat-square&color=lightgrey"/> -->
<!--     </a> -->
</p>

<p align="center">
    <p align="center">
        Visit the <a href="https://disent.dontpanic.sh/">docs</a> for more info, or browse the  <a href="https://github.com/nmichlo/disent/releases">releases</a>.
    </p>
</p>

----------------------

### Overview

Disent is a modular disentangled representation learning framework for auto-encoders, built upon pytorch-lightning. This framework consists of various composable components that can be used to build and benchmark disentanglement pipelines.

> The name of the framework is derived from both **disent**anglement and scientific **dissent**.

#### Goals

Disent aims to fill the following criteria:
- Provide **high quality**, **readable** and **easily comparable** implementations of VAEs
- Use **best practice** eg. `torch.distributions`
- Be extremely **flexible** & configurable

#### Citing Disent

Please use the following citation if you use Disent in your research:

```bibtex
@Misc{Michlo2021Disent,
  author =       {Nathan Juraj Michlo},
  title =        {Disent - A modular disentangled representation learning framework for pytorch},
  howpublished = {Github},
  year =         {2021},
  url =          {https://github.com/nmichlo/disent}
}
```

----------------------

### Getting Started

**WARNING**: Disent is still under active development. Features and APIs are not considered stable, but should be expected to change! A very limited set of tests currently exist which will be expanded upon in time.

The easiest way to use disent is by running `experiements/hydra_system.py` and changing the root config in `experiements/config/config.yaml`. Configurations are managed with [Hydra Config](https://github.com/facebookresearch/hydra)

**Pypi**:

1. Install with: `pip install disent` (This will most likely be outdated)

2. Visit the [docs](https://disent.dontpanic.sh)!

**Source**:

1. Clone with: `git clone --branch dev https://github.com/nmichlo/disent.git`

2. Change your working directory to the root of the repo: `cd disent`

3. Install the requirements for python 3.8 with `pip3 install -r requirements.txt` 

4. Run the default experiment after configuring `experiments/config/config.yaml`
   by running `PYTHONPATH=. python3 experiments/run.py`

----------------------

### Features

Disent includes implementations of modules, metrics and datasets from various papers. However modules marked with a "🧵" are newly introduced in disent for [nmichlo](https://github.com/nmichlo)'s MSc. research!

#### Frameworks
- **Unsupervised**:
  + [VAE](https://arxiv.org/abs/1312.6114)
  + [Beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl)
  + [DFC-VAE](https://arxiv.org/abs/1610.00291)
  + [DIP-VAE](https://arxiv.org/abs/1711.00848)
  + [InfoVAE](https://arxiv.org/abs/1706.02262)
  + [BetaTCVAE](https://arxiv.org/abs/1802.04942)
- **Weakly Supervised**:
  + [Ada-GVAE](https://arxiv.org/abs/2002.02886) *`AdaVae(..., average_mode='gvae')`* Usually better than the Ada-ML-VAE
  + [Ada-ML-VAE](https://arxiv.org/abs/2002.02886) *`AdaVae(..., average_mode='ml-vae')`*
- **Supervised**:
  + [TVAE](https://arxiv.org/abs/1802.04403)
- **Experimental**:
  + 🧵 Ada-TVAE
  + *various others not worth mentioning*

Many popular disentanglement frameworks still need to be added, please
submit an issue if you have a request for an additional framework.

<details><summary><b>todo</b></summary><p>

+ FactorVAE
+ GroupVAE
+ MLVAE

</p></details>

#### Metrics
- **Disentanglement**:
  + [FactorVAE Score](https://arxiv.org/abs/1802.05983)
  + [DCI](https://openreview.net/forum?id=By-7dz-AZ)
  + [MIG](https://arxiv.org/abs/1802.04942)
  + [SAP](https://arxiv.org/abs/1711.00848)
  + [Unsupervised Scores](https://github.com/google-research/disentanglement_lib)
  + 🧵 Flatness Score

Some popular metrics still need to be added, please submit an issue if you wish to
add your own or you have a request for an additional metric.

<details><summary><b>todo</b></summary><p>

+ [DCIMIG](https://arxiv.org/abs/1910.05587)
+ [Modularity and Explicitness](https://arxiv.org/abs/1802.05312)

</p></details>

#### Datasets:

Various common datasets used in disentanglement research are implemented, as well as new sythetic datasets that are generated programatically on the fly. These are convenient and lightweight, not requiring storage space.

- **Ground Truth**:
  + Cars3D
  + dSprites
  + MPI3D
  + SmallNORB
  + Shapes3D

- **Ground Truth Non-Overlapping (Synthetic)**:
  + 🧵 XYBlocks: *3 blocks of decreasing size that move across a grid. Blocks can be one of three colors R, G, B. if a smaller block overlaps a larger one and is the same color, the block is xor'd to black.*
  + 🧵 XYSquares: *3 squares (R, G, B) that move across a non-overlapping grid. Obervations have no channel-wise loss overlap.*
  + 🧵 XYObject: *A simplistic version of dSprites with a single square.*

  ##### Input Transforms + Input/Target Augmentations
  
  - Input based transforms are supported.
  - Input and Target CPU and GPU based augmentations are supported.

#### Schedules/Annealing:

Hyper-parameter annealing is supported through the use of schedules. The currently implemented schedules include:

- Linear Schedule
- [Cyclic](https://arxiv.org/abs/1903.10145) Schedule
- Cosine Wave Schedule
- *Various other wrapper schedules*

----------------------

### Why?
  
- Created as part of my Computer Science MSc scheduled for completion in 2021.

- I needed custom high quality implementations of various VAE's.

- A pytorch version of [disentanglement_lib](https://github.com/google-research/disentanglement_lib).

- I didn't have time to wait for [Weakly-Supervised Disentanglement Without Compromises](https://arxiv.org/abs/2002.02886) to release
  their code as part of disentanglement_lib. (As of September 2020 it has been released, but has unresolved [discrepencies](https://github.com/google-research/disentanglement_lib/issues/31)).

- disentanglement_lib still uses outdated Tensorflow 1.0, and the flow of data is unintuitive because of its use of [Gin Config](https://github.com/google/gin-config).

----------------------

### Architecture

**disent**
- `disent/data`: raw groundtruth datasets
- `disent/dataset`: dataset wrappers & sampling strategies
- `disent/framework`: frameworks, including Auto-Encoders and VAEs
- `disent/metrics`: metrics for evaluating disentanglement using ground truth datasets
- `disent/model`: common encoder and decoder models used for VAE research
- `disent/schedule`: annealing schedules that can be registered to a framework
- `disent/transform`: transform operations for processing & augmenting input and target data from datasets

**experiment**
- `experiment/run.py`: entrypoint for running basic experiments with [hydra](https://github.com/facebookresearch/hydra) config
- `experiment/config`: root folder for [hydra](https://github.com/facebookresearch/hydra) config files
- `experiment/util`: various helper code, pytorch lightning callbacks & visualisation tools for experiments

----------------------

### Example Code

The following is a basic working example of disent that trains a BetaVAE with a cyclic
beta schedule and evaluates the trained model with various metrics.

<details><summary><b>Basic Example</b></summary>
<p>

```python3
import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader
from disent.data.groundtruth import XYObjectData
from disent.dataset.groundtruth import GroundTruthDataset
from disent.frameworks.vae.unsupervised import BetaVae
from disent.metrics import metric_dci, metric_mig
from disent.model.ae import EncoderConv64, DecoderConv64, AutoEncoder
from disent.schedule import CyclicSchedule
from disent.transform import ToStandardisedTensor

# We use this internally to test this script.
# You can remove all references to this in your own code.
from disent.util import is_test_run

# create the dataset & dataloaders
# - ToStandardisedTensor transforms images from numpy arrays to tensors and performs checks
data = XYObjectData()
dataset = GroundTruthDataset(data, transform=ToStandardisedTensor())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# create the BetaVAE model
# - adjusting the beta, learning rate, and representation size.
module = BetaVae(
    make_optimizer_fn=lambda params: Adam(params, lr=5e-4),
    make_model_fn=lambda: AutoEncoder(
        # z_multiplier is needed to output mu & logvar when parameterising normal distribution
        encoder=EncoderConv64(x_shape=dataset.x_shape, z_size=6, z_multiplier=2),
        decoder=DecoderConv64(x_shape=dataset.x_shape, z_size=6),
    ),
    cfg=BetaVae.cfg(beta=0.004)
)

# cyclic schedule for target 'beta' in the config/cfg. The initial value from the
# config is saved and multiplied by the ratio from the schedule on each step.
# - based on: https://arxiv.org/abs/1903.10145
module.register_schedule('beta', CyclicSchedule(
    period=1024,  # repeat every: trainer.global_step % period
))

# train model
# - for 65536 batches/steps
trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_steps=65536, fast_dev_run=is_test_run())
trainer.fit(module, dataloader)

# compute disentanglement metrics
# - we cannot guarantee which device the representation is on
# - this will take a while to run
get_repr = lambda x: module.encode(x.to(module.device))

metrics = {
    **metric_dci(dataset, get_repr, num_train=10 if is_test_run() else 1000, num_test=5 if is_test_run() else 500, show_progress=True),
    **metric_mig(dataset, get_repr, num_train=20 if is_test_run() else 2000),
}

# evaluate
print('metrics:', metrics)
```

</p>
</details>

Visit the [docs](https://disent.dontpanic.sh) for more examples!

----------------------
