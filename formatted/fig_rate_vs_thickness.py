import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from common.util import chdir_root, ensure_folder_exists
from kinetics.plot_kinetics import plot_rate_vs_thickness


def main():
    chdir_root()
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    plt.rcParams['lines.linewidth'] = 1
    plt.figure(figsize=(4.5, 3))
    ax = plt.axes()
    plot_rate_vs_thickness(outpath=None)
    outpath = os.path.join('formatted', 'dist', 'Fig S11.pdf')
    ensure_folder_exists(outpath)
    plt.savefig(outpath)


if __name__ == "__main__":
    main()
