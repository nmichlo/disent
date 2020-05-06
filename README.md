# Disent

Disentanglement Library for pytorch and pytorch-lightning, based on
Google's [disentanglement_lib](https://github.com/google-research/disentanglement_lib), but more notebook friendly.

**DISCLAIMER:** This project is mostly a fork of disentanglement_lib and much of the code is not my own except with minor changes to make it work with pytorch, unfortunatly it didnt start out that way and is why the repo is not marked as such. I have left in the original copyrights above the files where appropriate.

## Another disentanglement library?
  
- I needed to become more familiar with VAE's

- I wanted pretty visualisations in my notebooks.

- [Weakly-Supervised Disentanglement Without Compromises](https://arxiv.org/abs/2002.02886) state they would release
  their code as part of disentanglement_lib but so far nothing has happened, and I wanted to play with their techniques.
  
- The original disentanglement_lib was written using Tensorflow 1, with
  Gin configurations controlling execution. I was not a fan.

## Usage

Disent is still under active development, and the API is subject to change at any time.

If you still wish to use this library, take a look at the various notebooks for examples.
