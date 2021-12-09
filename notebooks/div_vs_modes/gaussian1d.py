import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

class GaussianMixture1D:
    """
     - Multivariate distribution with multivariate (1D) Gaussian distribution components 
     - Each component defined by: means, stds
     - Mixture is defined by a vector of mixing proportions: mixture_prs
     
    Args:
    - mixture_prs: list mixture prs (applied to logpdf components)
    - means: list means of logpdf
    - stds: list std of logpdf


    Returns:
    - mixture_dbn: pdf(samples)
    """


    def __init__(self, mixture_prs, means, stds):
        self.num_mixtures = len(mixture_prs)
        self.mixture_prs = mixture_prs
        self.means = means
        self.stds = stds

    def sample(self, num_samples=1):
        """        
        samples num_samples accordung to mixture_prs, chooses associated means, stds & samples from normal
        Returns:
        list of num_samples nml pdf samples 
        """

        mixture_ids = np.random.choice(self.num_mixtures, size=num_samples, p=self.mixture_prs)
        result = np.zeros([num_samples])
        for sample_idx in range(num_samples):
            result[sample_idx] = np.random.normal(
                loc=self.means[mixture_ids[sample_idx]],
                scale=self.stds[mixture_ids[sample_idx]]
            )
        return result

    def logpdf(self, samples):
        mixture_logpdfs = np.zeros([len(samples), self.num_mixtures])
        for mixture_idx in range(self.num_mixtures):
            mixture_logpdfs[:, mixture_idx] = scipy.stats.norm.logpdf(
                samples,
                loc=self.means[mixture_idx],
                scale=self.stds[mixture_idx]
            )
        return sp.special.logsumexp(mixture_logpdfs + np.log(self.mixture_prs), axis=1)

    def pdf(self, samples):
        mixture_dbn = np.exp(self.logpdf(samples))
        return mixture_dbn