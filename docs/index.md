# Overview

PyTorchLightning disentanglement library implementing various VAEs
that can easily be run with Hydra Config.

Various other unique optional features exist, including data augmentations,
as well as the first (?) unofficial implementation of the tensorflow based [Ada-GVAE](https://github.com/google-research/disentanglement_lib).

## Goals

- Easy configuration and running of custom experiments.
- Flexible and composable components for vision based AE research.
- Challenge common opinions and assumptions in AE research through data.

[comment]: <> (  > Dissent: "Opinions at odds with those commonly held.")

## Disent Structure

The `disent` framework can be decomposed into various parts:

### data

Common and custom data for vision based AE, VAE and Disentanglement research.

- Most data is generated from ground truth factors which is necessary for evaluation using disentanglement metrics.
  Each image generated from ground truth data has the ground truth variables available.
  
??? example
    ```python
    --8<-- "docs/examples/overview_data.py"
    ```

### dataset

Wrappers for the aforementioned data. Ground truth variables of the data can be used to generate pairs or ordered sets for each observation in the datasets.

??? example "Examples"
    === "Singles Numpy Dataset"
        ```python3
        --8<-- "docs/examples/overview_dataset_single.py"
        ```
    === "Paired Tensor Dataset"
        ```python3
        --8<-- "docs/examples/overview_dataset_pair.py"
        ```
    === "Paired Augmented Tensor Dataset"
        ```python3
        --8<-- "docs/examples/overview_dataset_pair_augment.py"
        ```
    === "Paired Tensor Dataloader"
        ```python3
        --8<-- "docs/examples/overview_dataset_loader.py"
        ```


### frameworks

PytorchLightning modules that contain various AE or VAE implementations.

???+ example "Examples"
    === "AE"
        ```python3
        --8<-- "docs/examples/overview_framework_ae.py"
        ```
    === "BetaVAE"
        ```python3
        --8<-- "docs/examples/overview_framework_betavae.py"
        ```
    === "Ada-GVAE"
        ```python3
        --8<-- "docs/examples/overview_framework_adagvae.py"
        ```


### metrics

Various metrics used to evaluate representations learnt by AEs and VAEs.

??? example
    ```python3
    --8<-- "docs/examples/overview_metrics.py"
    ```

### schedules

Various hyper-parameter schedules can be applied if models reference
their config values directly. Such as `beta` (`cfg.beta`) in all the BetaVAE derived classes.

A warning will be printed if the hyper-parameter does not exist in the config, instead of crashing.

??? example
    ```python3
    --8<-- "docs/examples/overview_framework_betavae_scheduled.py"
    ```

### Datasets Without Ground-Truth Factors

Using datasets that do not have ground truth factors require custom wrappers with
custom sampling procedures, however metrics cannot be computed over them.

We can implement an MNIST example using the builtin random sampler.

??? example
    ```python3
    --8<-- "docs/examples/mnist_example.py"
    ```
