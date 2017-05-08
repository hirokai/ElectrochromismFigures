import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from common.data_tools import load_csv
from common.util import chdir_root, ensure_folder_exists


# http://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth
def smooth(x, window_len=11, window='hanning'):
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def list_files():
    fs = list()
    fs.append("/Users/hiroyuki/Google Drive/ExpData/Force gauge/20170424 PU only/01.csv")
    for i in range(1, 5):
        fs.append("/Users/hiroyuki/Google Drive/ExpData/Force gauge/20170426 PU only/%02d.csv" % i)
    return fs


def main():
    chdir_root()
    files = list_files()
    for i, csv_path in enumerate(files):
        vs = load_csv(csv_path, skip_rows=13, numpy=True)
        xs = vs[:, 1] / 15
        ys = vs[:, 0] / (1e-2 * 3.095e-6) / 1e6     # Thickness is from 2017/4/26 measurement by a surface profiler.
        winlen_half = 20
        ys = smooth(ys, window_len=winlen_half * 2 + 1)
        ys = ys[winlen_half:-winlen_half]
        if i == 0:
            ys = -ys
        if i >= 2:
            plt.plot(xs, ys, label=os.path.basename(csv_path))
    plt.xlim([0, 3])
    plt.ylim([0, 3])
    plt.legend()
    plt.show()


main()
