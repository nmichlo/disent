import numpy as np
#import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *
from gaussian1d import GaussianMixture1D

def main():

    #sample 5 p_means2 from [0, 10] (including boundaries)
    p_means2_min = 0
    p_means2_max = 10
    #num_q_means
    num_p_means2 = 5
    # list drives graphing process
    p_mean2_list = np.linspace(p_means2_min, p_means2_max, num_p_means2) 
    p = [None] * num_p_means2

    q_best_fwd = [None] * num_p_means2
    kl_best_fwd = [None] * num_p_means2
    q_best_rev = [None] * num_p_means2
    kl_best_rev = [None] * num_p_means2


    for p_mean2_idx, p_mean2 in enumerate(p_mean2_list):
        # mix with std nml 
        p_mixture_probs = np.array([0.5, 0.5])
        p_means = np.array([0, p_mean2])
        p_stds = np.array([1, 1])
        
        # array of 2 x 1D Guassian mixtures
        # function that is graphed
        p[p_mean2_idx] = GaussianMixture1D(p_mixture_probs, p_means, p_stds)
        

        # fitting around means of p
        q_means_min = np.min(p_means) - 1
        q_means_max = np.max(p_means) + 1
        num_q_means = 20
        q_means = np.linspace(q_means_min, q_means_max, num_q_means)
        
        # more variant std than p
        q_stds_min = 0.1
        q_stds_max = 5
        num_q_stds = 20
        q_stds = np.linspace(q_stds_min, q_stds_max, num_q_stds)

        # using trapz to get xs based on q's placed around p's we defined 
        trapz_xs_min = np.min(np.append(p_means, q_means_min)) - 3 * np.max(np.append(p_stds, q_stds_max))
        trapz_xs_max = np.max(np.append(p_means, q_means_min)) + 3 * np.max(np.append(p_stds, q_stds_max))
        num_trapz_points = 1000
        trapz_xs = np.linspace(trapz_xs_min, trapz_xs_max, num_trapz_points)

        # function that is graphed
        q_best_fwd[p_mean2_idx], kl_best_fwd[p_mean2_idx] = minimize_pq(
            p[p_mean2_idx], trapz_xs, q_means, q_stds
        )
        # function that is graphed
        q_best_rev[p_mean2_idx], kl_best_rev[p_mean2_idx] = minimize_qp(
            p[p_mean2_idx], trapz_xs, q_means, q_stds
        )

    # plotting
    fig, axs = plt.subplots(nrows=1, ncols=num_p_means2, sharex=True, sharey=True)
    fig.set_size_inches(24, 5)
    for p_mean2_idx, p_mean2 in enumerate(p_mean2_list):
        xs_min = -5
        xs_max = 15
        num_plot_points = 1000
        xs = np.linspace(xs_min, xs_max, num_plot_points)
        axs[p_mean2_idx].plot(xs, p[p_mean2_idx].pdf(xs), label='$p$', color='black')
        axs[p_mean2_idx].plot(xs, q_best_fwd[p_mean2_idx].pdf(xs), label='$\mathrm{argmin}_q \,\mathrm{KL}(p || q)$', color='black', linestyle='dashed')
        axs[p_mean2_idx].plot(xs, q_best_rev[p_mean2_idx].pdf(xs), label='$\mathrm{argmin}_q \,\mathrm{KL}(q || p)$', color='black', linestyle='dotted')

        axs[p_mean2_idx].spines['right'].set_visible(False)
        axs[p_mean2_idx].spines['top'].set_visible(False)
        axs[p_mean2_idx].set_yticks([])
        axs[p_mean2_idx].set_xticks([])

    axs[2].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0), fontsize='small')
    filenames = ['reverse_forward_kl.png'] #'reverse_forward_kl.pdf',
    for filename in filenames:
        fig.savefig(filename, bbox_inches='tight', dpi=200)
        print('Saved to {}'.format(filename))


if __name__ == '__main__':
    main()