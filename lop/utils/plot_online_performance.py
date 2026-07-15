import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


def generate_utility_histogram(
        distributions=None,
        labels=None,
        bins=100,
        xlabel='Utility score',
        ylabel='Density',
        fontsize=18,
        caption=None,
        filename='utility_histogram.png',
        svg=False,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    xmin = 1e-6  # avoid x=0 singularity
    xmax = 0.35

    # Plot histograms
    for idx, values in enumerate(distributions):
        label = labels[idx] if labels is not None else None

        ax.hist(
            values,
            bins=bins,
            density=True,
            range=(0.0, xmax),
            alpha=0.45,
            label=label,
        )

    # ------------------------------------------------------------------
    # Overlay normalized power law
    # ------------------------------------------------------------------
    x = np.linspace(xmin , xmax, 1000)
    power_law_alpha = -3

    shift = 0.5

    # Evaluate the power law at shifted coordinates
    p = (x + shift) ** power_law_alpha



    # Normalize over the plotted x-range
    p /= np.trapz(p, x)

    p /= p.sum()

    ax.plot(
        x,
        p,
        color='black',
        linewidth=2.5,
        linestyle='--',
        label=rf'Power law ($\alpha={power_law_alpha}$)'
    )

    # ------------------------------------------------------------------

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(0, xmax)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if labels is not None:
        ax.legend(fontsize=fontsize - 2)

    if caption is not None:
        ax.set_title(caption, fontsize=fontsize)

    plt.savefig(filename, bbox_inches='tight', dpi=500)
    plt.close()


def generate_online_performance_plot(
        performances=None,
        colors=None,
        xticks=[],
        xticks_labels=None,
        yticks=[],
        yticks_labels=None,
        m=20000,
        xlabel='',
        ylabel='',
        labels=None,
        caption=None,
        fontsize=24,
        log_scale_x=False,
        log_scale_y=False,
        svg=False,
):

    """
    This function plots the online performance of an algorithm for a various hyper parameter settings.
    It plots the mean and std-error for each configuration
    """
    shape = np.shape(performances)
    if colors is None:
        colors = [(1, 0, 0, 1), (0.5, 0.5, 0, 1), (0, 1, 0, 1)]
    fig, ax = plt.subplots()

    for index_hyper_param_label in range(shape[0]):
        x = np.array([i for i in range(shape[-1])])
        mean = np.mean(performances[index_hyper_param_label], axis=0)
        num_samples = np.shape(performances)[1]
        std_err = np.std(performances[index_hyper_param_label], axis=0)/sqrt(num_samples)
        label = ''
        if labels is not None:
            label = labels[index_hyper_param_label]
        color = colors[index_hyper_param_label]
        plt.plot(x*m, mean, '-', label=label, color=color)
        plt.fill_between(x*m, mean - std_err, mean + std_err, color=color, alpha=0.2)

    # h.set_rotation(0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if xticks_labels is None:
        xticks_labels = xticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels, fontsize=fontsize)
    if len(yticks) > 0:
        ax.set_yticks(yticks)
        ax.set_ylim(yticks[0], yticks[-1])
    if yticks_labels is not None:
        ax.set_yticklabels(yticks_labels, fontsize=fontsize)
    elif len(yticks) > 0:
        ax.set_yticklabels(yticks, fontsize=fontsize)
        ax.set_ylim(yticks[0], yticks[-1])

    if log_scale_y:
        ax.set_yscale('log')
    if log_scale_x:
        ax.set_xscale('log')
    ax.yaxis.grid()

    if labels is not None:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if caption is not None:
        ax.set_title(caption)
    if svg:
        plt.savefig('comparison.svg', bbox_inches='tight', dpi=500)
    else:
        plt.savefig('comparison.png', bbox_inches='tight', dpi=500)
    plt.close()
