import matplotlib.pyplot as plt
from common.data_tools import load_csv
from common.util import chdir_root
import numpy as np
from common.figure_tools import colors10
import os


def run_set(corrected=False):
    chdir_root()
    for i in range(1, 9):
        plt.subplot(8, 2, 1 + 2 * (i - 1) + (1 if corrected else 0))
        path = "data/100cycles/%s/20170502/Movie %02d.csv" % ("corrected" if corrected else "raw", i)
        vs = load_csv(path, numpy=True)
        if vs is not None:
            ys = vs[:, 0]
            ts = np.array(range(len(vs)))
            plt.plot(ts, ys, c=colors10[1] if corrected else colors10[0])
            plt.xlim([0, 600])
            plt.ylim([0, 60])


cycles_initial = [0, 10, 20]


def run_set2(corrected=False):
    chdir_root()
    count = 0
    for i in [1, 3, 5, 7, 8]:
        count += 1
        plt.subplot(2, 5, count + (5 if corrected else 0))
        path = "data/100cycles/%s/20170502/Movie %02d.csv" % ("corrected" if corrected else "raw", i)
        vs = load_csv(path, numpy=True)
        if vs is not None:
            ys = vs[:, 0]
            if corrected:
                mx = np.max(ys[100:201])
                mn = np.min(ys[100:201])
                print(mx, mn, mx - mn)
            ts = np.array(range(len(vs)))
            plt.plot(ts, ys, c=colors10[1] if corrected else colors10[0])
            plt.xlim([0, 600])
            plt.ylim([0, 60])


def main():
    chdir_root()
    plt.figure(figsize=(20, 10))
    run_set2()
    run_set2(corrected=True)
    plt.savefig(os.path.join("dist", "fig 1200 cycles.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
