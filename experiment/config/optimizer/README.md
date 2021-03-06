# Optimizer Info

## Descending through a Crowded Valley -- Benchmarking Deep Learning Optimizers
https://arxiv.org/abs/2007.01547

**Best Optimizers On Average**:
- Adam: `torch.optim.Adam`
- RAdam: `torch_optimizer.RAdam`
- Nadam: `# no official pytorch implementation #`
- AMSGrad: `torch.optim.Adam(amsgrad=True)`
- RMSProp: `torch.optim.RMSprop` (kinda slow convergence, from my tests, and spikes)
The rest have bad score on at least one of the benchmarks.

**tl;dr** You cant really go wrong with any of the above! (Even from my own tests)


## My Own Tests
**Disclaimer**: I by no means have much experience with optimizers and tuning optimizer parameters.
                This is just what I found seems to work best on average with minimal tuning.

**tl;dr:** if not using a lr schedule (investigate) stick to:
   - adam or amsgrad for lower variance.
   - radam for best performance.
   - yogi or adabelief are possibilities too and arent much worst than the others over a long run, maybe worse/better initially, but with more variance initially (not at the end it seems).
   - Extensive parameter tuning not done for other methods (sgd, adabound, amsbound), these performed badly and didnt converge with default settings.


With default values (lr = 1e-3):
- **bad in general** (possibly needs a lr schedule?):
  - sgd
  - adabound
  - amsbound
  
- **decent**
  - **1** (1x1) radam (higher variance for adavae but **better scores**, lower variance for beta but same scores) (sometimes decays if lr is too high or if plateaued, but generally the mean perfromance is good)
  - **6** (2x3) adabelief (slower/high variance initially but seems to catch up)
  - **8** (4x2) RMSProp (**very** slow initial convergence, and **spikes** often)
  - **15** (3x5) adam (generally quite low variance for both beta and adavae)
  - **24** (6x4) yogi \[default lr is actually 1e-2\]
  - **30** (5x6) amsgrad (generally quite low variance for both beta and adavae)

AdaVAE:
 1. radam (is generally best by a large margin, but higher variance)
 2. adabelief (is generally slower initially, but converges to adam and amsgrad or might be better)
 3. adam (variance is generally low)
 4. RMSProp (has very slow initial convergence, much slower than others, and spikes often which is odd)
 5. amsgrad (variance is generally slightly lower)
 6. yogi (is generally slower initially, but converges to adam and amsgrad  or better)

BetaVAE:
 1. radam (is generally best by a small margin, low variance compared to adavae and other methods)
 2. RMSProp has very slow initial convergence (much slower than others), and spikes often which is odd.
 3. adabelief (best initially but same as the rest after this, variance slightly high)
 4. yogi (is generally slower initially with high initial variance, but converges to adam and amsgrad or better)
 5. adam (variance is generally low)
 6. amsgrad (variance is generally slightly lower)

