import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from common.util import chdir_root, ensure_folder_exists
from film_thickness.plot_thickness import plot_thickness_rpm_multi


def main():
    raise "Stub!"
    chdir_root()
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    plt.figure(figsize=(6, 4.5))
    outpath = os.path.join('formatted', 'dist', 'Fig S5.pdf')
    ensure_folder_exists(outpath)
    plt.savefig(outpath)


if __name__ == "__main__":
    main()
