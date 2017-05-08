import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from common.data_tools import load_csv
from common.figure_tools import colors10
from common.util import chdir_root, ensure_folder_exists
import numpy as np

path = "data/sem-edx/20170428 SEM-EDX line scan - area 4.csv"


def main():
    chdir_root()
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    vs = load_csv(path, skip_rows=1, universal_new_line=True)
    vs = np.array([row[0:9] for row in vs])
    xs = vs[:, 0]
    ys_sulfur = vs[:, 8]
    ys_oxygen = vs[:, 2]
    ys_carbon = vs[:, 1]

    plt.figure(figsize=(6,4.5))

    plt.subplot(2,1,1)
    plt.plot(xs, ys_carbon,label="C",c=colors10[0])
    plt.plot(xs, ys_oxygen,label="O",c=colors10[1])
    plt.plot(xs, ys_sulfur,label="S",c=colors10[2])
    plt.xlim([0,4])
    plt.ylim([0,100])
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(xs, ys_sulfur,label="S",c=colors10[2])
    plt.xlim([0,4])
    plt.ylim([0,5])
    plt.legend()

    outpath = os.path.join('formatted', 'dist', 'Fig S4b.pdf')
    ensure_folder_exists(outpath)
    plt.savefig(outpath)
    plt.show()


if __name__ == "__main__":
    main()
