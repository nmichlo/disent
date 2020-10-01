# 🧶 Disent

Disentanglement Library for pytorch and pytorch-lightning. With an easy to use configuration based on Hydra.



## Another disentanglement library?
  
- I needed to become more familiar with VAE's (Currently working on my masters)

- **DISCLAIMER:** This project has its roots in the tensorflow [disentanglement_lib](https://github.com/google-research/disentanglement_lib) library.

- [Weakly-Supervised Disentanglement Without Compromises](https://arxiv.org/abs/2002.02886) stated they would release
  their code as part of disentanglement_lib... I didn't have time to wait... As of September it has been released.
  
- The disentanglement_lib still uses Tensorflow 1.0 and [Gin Config](https://github.com/google/gin-config) controls execution, **hiding** the flow of data in the library (I am not a fan).



## Features

### Frameworks
- **Unsupervised**:
  - <ins>VAE</ins>:
  - <ins>BetaVAE</ins>:
  - <ins>DFCVAE</ins>:
- **Weakly Supervised**:
    - <ins>Ada-GVAE</ins>: *`AdaVae(..., average_mode='gvae')`*
    - <ins>Ada-ML-VAE</ins>: *`AdaVae(..., average_mode='ml-vae')`*
- **Supervised**:
    - <ins>TVAE</ins>:

### Metrics
- **Disentanglement**:
    - <ins>FactorVAE score</ins>:
    - <ins>DCI</ins>:

### Datasets:
- **Ground Truth**:
    - <ins>Cars3D</ins>:
    - <ins>dSprites</ins>:
    - <ins>MPI3D</ins>:
    - <ins>SmallNORB</ins>:
    - <ins>Shapes3D</ins>:
- **Ground Truth Non-Overlapping (Synthetic)**:
    - <ins>XYBlocks</ins>: *3 blocks of decreasing size that move across a grid. Blocks can be one of three colors R, G, B. if a smaller block overlaps a larger one and is the same color, the block is xor'd to black.*
    - <ins>XYSquares</ins>: *3 squares (R, G, B) that move across a non-overlapping grid. Obervations have no channel-wise loss overlap.*
    - <ins>XYObject</ins>: *A simplistic version of dSprites with a single square.*



## Usage

Disent is still under active development (I an sorry there are no tests yet).

The easiest way to use this library is by running `experiements/hydra_system.py` and changing the config in `experiements/config/config.yaml`. Configurations are managed by [Hydra Config](https://github.com/facebookresearch/hydra)
