# Disent

PyTorch Lightning disentanglement framwork implementing various modular VAEs.

Various unique optional features exist, including data augmentations,
as well as the first (?) unofficial implementation of the tensorflow based [Ada-GVAE](https://github.com/google-research/disentanglement_lib).

## Goals

Disent aims to fill the following criteria:
1. Provide **high quality**, **readable**, **consistent** and **easily comparable** implementations of frameworks
2. **Highlight difference** between framework implementations by overriding **hooks** and minimising duplicate code 
3. Use **best practice** eg. `torch.distributions`
4. Be extremely **flexible** & configurable
5. Load data from disk for low memory systems

## Citing Disent

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
