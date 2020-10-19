# Optimizer Info

**Disclaimer**: I by no means have much experience with optimizers and tuning optimizer parameters.

With default values (lr = 1e-3):
- **bad in general** (possibly needs a lr schedule?):
  - sgd
  - adabound
  - amsbound
  
- **decent**
  1. adabelief (slower/high variance initially but seems to catch up)
  2. yogi \[default lr is actually 1e-2\]

- **good**
  1. radam (higher variance for adavae but better scores, lower variance for beta but sameish scores) (sometimes decays if lr is too high or if plateaued, but generally the mean perfromance is good)
  2. adam (generally quite low variance for both beta and adavae)
  3. amsgrad (generally quite low variance for both beta and adavae)
  
  
AdaVAE:
 1. radam is generally best by a large margin (but higher variance)
 2. adam (variance is generally low)
 3. amsgrad (variance is generally slightly lower)
 4. adabelief is generally slower initially, but converges to adam and amsgrad or might be better
 5. yogi is generally slower initially, but converges to adam and amsgrad  or better

BetaVAE:
 1. radam is generally best by a small margin (low variance compared to adavae and other methods)
 2. adam (variance is generally low)
 3. amsgrad (variance is generally slightly lower)
 4. adabelief (best initially but same as the rest after this, variance slightly high)
 5. yogi is generally slower initially with high initial variance, but converges to adam and amsgrad or better

