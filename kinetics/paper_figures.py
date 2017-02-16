import os
import numpy as np

from common.data_tools import load_csv
from figure_tools import colors10
from common.util import chdir_root, ensure_folder_exists

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns


def first_order(t0, k, a_i, a_f, t):
    y = a_i + (a_f - a_i) * (1 - np.exp(-k * (t - t0)))
    return y


def plot_fitting_curve(voltage, t_offset, color=colors10[0]):
    in_path = os.path.join('data', 'kinetics', 'fitted_manual', '20160512-13', '20 perc PEDOT - 2000 rpm',
                           'ox %s.csv' % voltage)
    t0, kinv, li, lf = map(float, load_csv(in_path)[0])

    ts = np.array(range(60))
    ys = [first_order(t0, 1.0 / kinv, li, lf, t + t_offset) for t in ts]

    plt.plot(ts, ys, c=color)


def main():
    chdir_root()
    plt.figure(figsize=(6, 4))

    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    for i, voltage in enumerate(['0.2', '0.4', '0.6', '0.8']):
        vs = load_csv(
            os.path.join('data', 'kinetics', 'split', '20160512-13', '20 perc PEDOT - 2000 rpm', 'ox %s.csv' % voltage),
            numpy=True)
        xs = np.array(range(vs.shape[0]))[4:]
        ys = vs[:, 1][4:]
        t_offset = xs[0]
        xs -= t_offset
        plt.scatter(xs, ys, c=colors10[i], edgecolor='none', s=15)
        plot_fitting_curve(voltage, t_offset, color=colors10[i])

    plt.xlim([0, 40])
    plt.ylim([30, 50])
    ax = plt.axes()

    major_locator = MultipleLocator(10)
    minor_locator = MultipleLocator(2)
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)

    major_locator = MultipleLocator(10)
    minor_locator = MultipleLocator(2)
    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_minor_locator(minor_locator)

    plt.xlabel('Time [sec]')
    plt.ylabel('L* value')

    outpath = os.path.join('kinetics', 'dist', 'kinetics_voltages.pdf')
    ensure_folder_exists(outpath)
    plt.savefig(outpath)
    plt.show()


if __name__ == "__main__":
    main()
