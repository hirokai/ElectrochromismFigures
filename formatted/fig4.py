import os
import matplotlib.pyplot as plt
import seaborn as sns

from common.util import chdir_root, ensure_folder_exists
from kinetics.paper_figures import plot_kinetics_voltage_dependence
from kinetics.plot_kinetics import plot_final_colors, plot_final_colors_predefined_dates, plot_rate_constants
from kinetics.plot_kinetics_space import plot_cie_space


def main():
    chdir_root()
    plt.figure(figsize=(8, 6))
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    plt.rcParams['lines.linewidth'] = 1

    ax = plt.subplot(2, 2, 1)
    plot_kinetics_voltage_dependence(outpath=None, ax=ax)

    ax = plt.subplot(2, 2, 2)
    plot_final_colors_predefined_dates(ax=ax,normalize=True)
    plt.ylabel("L* value normalized to 0.5 um thickness")
    plt.ylim([10,60])
    ax.legend_.remove()

    ax = plt.subplot(2, 2, 3)
    plot_rate_constants()

    ax = plt.subplot(2, 2, 4)
    plot_cie_space(ax=ax)

    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)

    outpath = os.path.join('formatted', 'dist', 'Fig 4.pdf')
    ensure_folder_exists(outpath)
    plt.savefig(outpath)
    plt.show()

    plt.figure(figsize=(6, 4.5))
    plot_final_colors_predefined_dates(ax=ax,normalize=False)
    outpath2 = os.path.join('formatted', 'dist', 'Fig S7.pdf')
    ensure_folder_exists(outpath2)
    plt.savefig(outpath2)
    plt.show()


if __name__ == "__main__":
    main()
