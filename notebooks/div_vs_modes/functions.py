import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from gaussian1d import GaussianMixture1D

def approx_kl(gmm_p, gmm_q, xs):
    """
    Approximates KL Divergence 
    
    for x in xs:
        D_kl = ∫p(x)log(p(x)/q(x))dx

    """
    ys = gmm_p.pdf(xs) * (gmm_p.logpdf(xs) - gmm_q.logpdf(xs))
    # using np.trapz to approx intergration
    D_kl = np.trapz(ys, xs)
    return D_kl 


def minimize_pq(p, xs, q_means, q_stds):
    """
    Choose:
        [q_mean_best, q_std_best] from [q_means, q_stds] s.t. min D(p,q)_kl 
    """

    q_mean_best = None
    q_std_best = None
    kl_best = np.inf
    for q_mean in q_means:
        for q_std in q_stds:
            q = GaussianMixture1D(np.array([1]), np.array([q_mean]), np.array([q_std]))
            kl = approx_kl(p, q, xs)
            if kl < kl_best:
                kl_best = kl
                q_mean_best = q_mean
                q_std_best = q_std

    q_best = GaussianMixture1D(np.array([1]), np.array([q_mean_best]), np.array([q_std_best]))
    return q_best, kl_best


def minimize_qp(p, xs, q_means, q_stds):
    """
    Same as above with reverse KL
    Choose:
    [q_mean_best, q_std_best] from [q_means, q_stds] s.t. min D(q,p)_kl 
    """
    q_mean_best = None
    q_std_best = None
    kl_best = np.inf
    for q_mean in q_means:
        for q_std in q_stds:
            q = GaussianMixture1D(np.array([1]), np.array([q_mean]), np.array([q_std]))
            kl = approx_kl(q, p, xs)
            if kl < kl_best:
                kl_best = kl
                q_mean_best = q_mean
                q_std_best = q_std

    q_best = GaussianMixture1D(np.array([1]), np.array([q_mean_best]), np.array([q_std_best]))
    return q_best, kl_best

def minimize_fdiv(p, xs, q_means, q_stds, beta, gamma):
    """
    Same as above with fdiv =  beta * fwd_kl + gamma * bckwrd_kl 
    Choose:
    [q_mean_best, q_std_best] from [q_means, q_stds] s.t. min D(q,p)_kl 
    """
    q_mean_best = None
    q_std_best = None
    fdiv_best = np.inf
    for q_mean in q_means:
        for q_std in q_stds:
            q = GaussianMixture1D(np.array([1]), np.array([q_mean]), np.array([q_std]))
            fdiv = beta * approx_kl(q, p, xs) + gamma * approx_kl(p, q, xs)
            if fdiv < fdiv_best:
                fdiv_best = fdiv
                q_mean_best = q_mean
                q_std_best = q_std

    q_best = GaussianMixture1D(np.array([1]), np.array([q_mean_best]), np.array([q_std_best]))
    return q_best, fdiv_best