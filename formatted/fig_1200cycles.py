import matplotlib.pyplot as plt
from common.data_tools import load_csv
from common.util import chdir_root
import numpy as np
from common.figure_tools import colors10
import os


def run_set():
    chdir_root()
    count = 0
    contrast = []
    for i in [1, 3, 5, 7, 8]:
        count += 1
        plt.subplot(1, 5, count)
        path = "data/100cycles/corrected/20170502/Movie %02d.csv" % i
        vs = load_csv(path, numpy=True)
        if vs is not None:
            ys = vs[:, 0]
            mx = np.max(ys[100:201])
            mn = np.min(ys[100:201])
            contrast.append([mx, mn, mx - mn])
            ts = np.array(range(len(vs)))
            plt.plot(ts, ys, c=colors10[0])
            plt.xticks(np.arange(0, 600, 60.0))
            plt.title(["", "0 cycles", "220 cycles", "400 cycles", "600 cycles", "1200 cycles"][count])
            plt.xlim([0, 600])
            plt.ylim([0, 60])
    return np.array(contrast)


def abs_from_l(a):
    # L = -80.828597 * (Abs) + 56.321919
    return (a-56.321919)/(-80.828597)

def main():
    chdir_root()
    plt.figure(figsize=(12, 3))
    l_contrast = run_set()
    plt.savefig(os.path.join("formatted", "dist", "Fig S5a.pdf"))
    plt.show()
    # print(l_contrast)
    mn = l_contrast[:, 1]
    mx = l_contrast[:, 0]
    xs = [0, 220, 400, 600, 1200]
    plt.figure(figsize=(8,4))
    plt.subplot(2,1,1)
    plt.scatter(xs, mn, c=colors10[0], s=50, lw=0)
    plt.scatter(xs, mx, c=colors10[1], s=50, lw=0)
    plt.xlim([-10,1210])
    plt.ylim([0,60])
    plt.subplot(2,1,2)
    abs_diff = l_contrast[:,2] / 80.828597
    transmittance_contrast = np.power(10,abs_diff)
    plt.plot(xs, transmittance_contrast, c=colors10[2],marker='o',markersize=8,mew=0)
    print(transmittance_contrast)
    plt.xlim([-10,1210])
    plt.ylim([0.9,2.1])
    plt.yticks([1,1.5,2])
    plt.savefig(os.path.join("formatted", "dist", "Fig S5b.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
