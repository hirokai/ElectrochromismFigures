#
# Plotting final L values.
#

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

from kinetics import Kinetics, KineticsDataType, read_kinetics, read_final_l_predefined_dates
from figure_tools import colors10
from common.util import chdir_root, ensure_folder_exists
from matplotlib.ticker import MultipleLocator


def l_to_abs(l):
    # L = -80.828597 * (Abs) + 56.321919
    return (l - 56.321919) / (-80.828597)


def abs_to_l(a):
    # L = -80.828597 * (Abs) + 56.321919
    return -80.828597 * a + 56.321919


# Normalize L for [point] um-thick film.
def normalize_l(l, thickness, point=1):
    # thickness: in um
    # L: L* value (0-100)
    a = l_to_abs(l)
    a2 = a / thickness * point
    l2 = abs_to_l(a2)
    print("normalize_l(): %.1f -> %.3f -> %.3f -> %.1f" % (l, a, a2, l2))
    return l2


thickness_table = None


def load_thickness_table():
    global thickness_table
    thickness_table = {}
    with open(os.path.join("data", "thickness_summary.csv"), "rb") as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            pedot = int(row[0])
            rpm = int(float(row[1]))
            if pedot not in thickness_table:
                thickness_table[pedot] = {}
            thickness_table[pedot][rpm] = float(row[2])
    print(thickness_table)


def get_thickness(pedot, rpm):
    global thickness_table
    if thickness_table is None:
        load_thickness_table()
    return thickness_table[pedot][rpm]


def plot_series(dat, variable, pedot, rpm, mode, voltage, color=colors10[0], label=None, show=False, normalize=False):
    assert isinstance(dat, Kinetics)
    if variable == 'pedot':
        ps = [20, 30, 40, 60, 80]
        d = {p: dat.get_data(p, rpm, mode, voltage) for p in ps}
    elif variable == 'rpm':
        # rs = [500, 1000, 2000, 3000, 4000, 5000]
        rs = [500, 1000, 5000]
        d1 = {r: dat.get_data(pedot, r, mode, voltage) for r in rs}
        th = {500: 5.4, 1000: 4.2, 2000: 3.0, 3000: 2.8, 4000: 2.9, 5000: 2.8}  # 30perc PEDOT
        # th = {500: 7.0, 1000: 4.2, 2000: 3.7, 3000: 4.0, 4000: 3.1, 5000: 2.9}  # 20perc PEDOT
        d = {}
        print(d1)
        for k, v in d1.iteritems():
            d[th[k]] = v
    elif variable == 'mode':
        ms = ['const', 'ox', 'red']
        d = {m: dat.get_data(pedot, rpm, m, voltage) for m in ms}
    elif variable == 'voltage':
        vs = [-0.5, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
        if mode:
            d = {v: dat.get_data(pedot, rpm, mode, v) for v in vs}
        else:
            d = {v: dat.get_data(pedot, rpm, 'ox', v) or dat.get_data(pedot, rpm, 'red', v) for v in vs}
            # d = {v: dat.get_data(pedot, rpm, 'ox' if v > 0 else 'red', v) for v in vs}
    if normalize:
        d = {k: map(lambda l2: normalize_l(l2, get_thickness(pedot, rpm), point=0.5), v) for k, v in d.items()}
    ks = sorted(d.keys())
    vs = [np.mean(d[k] or []) for k in ks]
    values = {k: d[k] or [] for k in ks}
    l = [(zip([k] * 100, d[k]) or []) for k in ks if d[k] is not None]
    xlabels = {'voltage': 'Voltage [V]', 'pedot': 'PEDOT ratio [wt%]', 'rpm': 'Spin coating speed [rpm]'}
    if l:
        kv_all = reduce(lambda a, b: a + b, l)
        k_all = [a[0] for a in kv_all]
        v_all = [a[1] for a in kv_all]
        # print(kv_all)
        es = [np.std(d[k] or []) for k in ks]
        print('kinetics.plot_series(): voltage', voltage, zip(ks, vs, es), values)
        plt.errorbar(ks, vs, es, c=color, label=label)
        # plt.scatter(k_all, v_all, c=color)
        plt.title('%d perc PEDOT, %d rpm, %s, %.1f V' % (pedot or -1, rpm or -1, mode or '--', voltage or -1))
        plt.xlabel(xlabels.get(variable) or variable)
        plt.ylabel('Rate constant [sec^-1]')
        if show:
            plt.show()


# if outpath is None, the graph is for paper figures (using subplots).
def plot_final_colors(dates, outpath=None, ax=None):
    dat = read_kinetics(KineticsDataType.FinalL, dates=dates)
    if outpath is not None:
        plt.figure(figsize=(6, 4))
        ax = plt.axes()
    for i, pedot in enumerate([20, 40, 60, 80]):
        plot_series(dat, 'voltage', pedot, 2000, None, None, color=colors10[i], label='%d perc PEDOT' % pedot)
    plt.title('Varied PEDOT and voltage, ox, 2000 rpm')
    plt.ylabel('Final L value')

    major_locator = MultipleLocator(0.4)
    minor_locator = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)

    plt.legend(loc='lower right')

    if outpath is not None:
        ensure_folder_exists(outpath)
        plt.savefig(outpath)
        plt.show()


# if outpath is None, the graph is for paper figures (using subplots).
def plot_final_colors_predefined_dates(outpath=None, ax=None, normalize=False):
    dat = read_final_l_predefined_dates()
    if outpath is not None:
        plt.figure(figsize=(6, 4))
        ax = plt.axes()
    for i, pedot in enumerate([20, 40, 60, 80]):
        plot_series(dat, 'voltage', pedot, 2000, None, None, color=colors10[i], label='%d perc PEDOT' % pedot,
                    normalize=normalize)
    plt.title('Varied PEDOT and voltage, ox, 2000 rpm')
    plt.ylabel('Final L value')

    major_locator = MultipleLocator(0.4)
    minor_locator = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    major_locator = MultipleLocator(10)
    minor_locator = MultipleLocator(5)
    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_minor_locator(minor_locator)

    plt.legend(loc='lower right')

    if outpath is not None:
        ensure_folder_exists(outpath)
        plt.savefig(outpath)
        plt.show()


# if outpath is None, the graph is for paper figures (using subplots).
def plot_rate_constants(outpath=None):
    dat = read_kinetics(KineticsDataType.RateConstant, dates=None)
    if outpath is not None:
        plt.figure(figsize=(6, 4))
    for i, voltage in enumerate([0.2, 0.4, 0.6, 0.8]):
        plot_series(dat, 'pedot', None, 2000, 'ox', voltage, color=colors10[i], label='%.1f V' % voltage)
    plt.title('Varied PEDOT and voltage, ox, 2000 rpm')
    plt.xlim([0, 100])
    plt.ylim([0, 1.2])
    if outpath is not None:
        plt.legend()
    if outpath is not None:
        ensure_folder_exists(outpath)
        plt.savefig(outpath)
        plt.show()


# if outpath is None, the graph is for paper figures (using subplots).
def plot_rate_constants_reduction(outpath=None):
    dat = read_kinetics(KineticsDataType.RateConstant, dates=None)
    if outpath is not None:
        plt.figure(figsize=(6, 4))
    for i, pedot in enumerate([20, 40, 60, 80]):
        plot_series(dat, 'voltage', pedot, 2000, 'red', None, color=colors10[i], label='%d wt%% PEDOT' % pedot)
    plt.title('Varied PEDOT and voltage, red, 2000 rpm')
    plt.xlim([-0.6, 0.2])
    plt.ylim([0, 0.2])
    plt.legend()
    if outpath is not None:
        ensure_folder_exists(outpath)
        plt.savefig(outpath)
        plt.show()


def plot_rate_vs_thickness(outpath):
    dat = read_kinetics(KineticsDataType.RateConstant, dates=None)
    if outpath is not None:
        plt.figure(figsize=(6, 4))
    for i, pedot in enumerate([30]):
        # for i, pedot in enumerate([20, 30, 40, 60, 80]):
        plot_series(dat,
                    'rpm', pedot, None, 'ox', 0.8, color=colors10[i], label='%d perc PEDOT' % pedot)
    plt.title('Rate versus thickness')
    plt.legend(loc='upper right')
    plt.ylim([0, 1.2])
    plt.xlim([0, 8])
    plt.xlabel('Thickness [um]')

    if outpath is not None:
        ensure_folder_exists(outpath)
        plt.savefig(outpath)
        plt.show()


def main():
    chdir_root()

    # dates = None
    dates = ['20160512-13', '20161019']

    plot_rate_vs_thickness(outpath=os.path.join('kinetics', 'dist', '20170214 rate vs thickness.pdf'))

    plot_rate_constants(outpath=os.path.join('kinetics', 'dist', '20170214 rate pedot voltage.pdf'))

    plot_final_colors(dates=dates, outpath=os.path.join('kinetics', 'dist', '20170214 final l values.pdf'))


if __name__ == "__main__":
    main()
