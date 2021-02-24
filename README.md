
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
    <a href="https://github.com/nmichlo/disent">
        <img alt="last commit" src="https://img.shields.io/github/last-commit/nmichlo/disent?style=flat-square&color=lightgrey"/>
    </a>
</p>

<p align="center">
    <p align="center">
        Visit the <a href="https://disent.dontpanic.sh/">docs</a> for more info.
    </p>
</p>

----------------------

### Overview

Disent is a modular disentangled representation learning framework for auto-encoders, built upon pytorch-lightning, with its early roots in Google's tensorflow based [disentanglement-lib](https://github.com/google-research/disentanglement_lib). This framework consists of various composable components that can be used to build and benchmark disentanglement pipelines.

> The name of the framework is derived from both **disent**anglement and scientific **dissent**.

### Citing Disent

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

### Features

Disent includes implementations of modules, metrics and datasets from various papers. However modules marked with a "🧵" are newly introduced in disent for [nmichlo](https://github.com/nmichlo)'s MSc. research!

#### Frameworks
- **Unsupervised**:
  + [VAE](https://arxiv.org/abs/1312.6114)
  + [beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl)
  + [DFC-VAE](https://arxiv.org/abs/1906.01984)
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

- **todo**:
  + FactorVAE
  + InfoVAE
  + BetaTCVAE
  + DIPVAE
  + GroupVAE
  + MLVAE

#### Metrics
- **Disentanglement**:
  + [FactorVAE Score](https://arxiv.org/abs/1802.05983)
  + [DCI](https://openreview.net/forum?id=By-7dz-AZ)
  + [MIG](https://arxiv.org/abs/1802.04942)
  + [SAP](https://arxiv.org/abs/1711.00848)
  + Unsupervised Scores
  + 🧵 Flatness Score

Some popular metrics still need to be added, please submit an issue if you wish to
add your own or you have a request for an additional metric.
- **todo**:
  + [DCIMIG](https://arxiv.org/abs/1910.05587)
  + [Modularity and Explicitness](https://arxiv.org/abs/1802.05312)


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



### Why?
  
- Created as part of my Computer Science MSc scheduled for completion in 2021.

- I needed custom high quality implementations of various VAE's.

- A pytorch version of [disentanglement_lib](https://github.com/google-research/disentanglement_lib).

- I didn't have time to wait for [Weakly-Supervised Disentanglement Without Compromises](https://arxiv.org/abs/2002.02886) to release
  their code as part of disentanglement_lib. (As of September 2020 it has been released, but has unresolved [discrepencies](https://github.com/google-research/disentanglement_lib/issues/31)).

- disentanglement_lib still uses outdated Tensorflow 1.0, and the flow of data is unintuitive because of its use of [Gin Config](https://github.com/google/gin-config).
