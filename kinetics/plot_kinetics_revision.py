#
# Plotting final L values.
#

import os
import matplotlib.pyplot as plt
import numpy as np

from kinetics import Kinetics, KineticsDataType, read_kinetics
from common.data_tools import colors10


def plot_series(dat, variable, pedot, rpm, mode, voltage, color=colors10[0], label=None, show=False):
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
    ks = sorted(d.keys())
    vs = [np.mean(d[k] or []) for k in ks]
    l = [(zip([k] * 100, d[k]) or []) for k in ks if d[k] is not None]
    if l:
        kv_all = reduce(lambda a, b: a + b, l)
        k_all = [a[0] for a in kv_all]
        v_all = [a[1] for a in kv_all]
        print(kv_all)
        es = [np.std(d[k] or []) for k in ks]
        plt.errorbar(ks, vs, es, c=color, label=label)
        plt.scatter(k_all, v_all, c=color)
        plt.title('%d perc PEDOT, %d rpm, %s, %.1f V' % (pedot or -1, rpm or -1, mode or '--', voltage or -1))
        plt.xlabel(variable)
        plt.ylabel('Rate constant [sec^-1]')
        if show:
            plt.show()


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))

    dat = read_kinetics(KineticsDataType.RateConstant)
    plt.figure(figsize=(6, 4))
    # plt.figure(figsize=(10, 10))
    for i, pedot in enumerate([30]):
        # for i, pedot in enumerate([20, 30, 40, 60, 80]):
        plot_series(dat,
                    'rpm', pedot, None, 'ox', 0.8, color=colors10[i], label='%d perc PEDOT' % pedot)
    # plt.title('Varied PEDOT and rpm, ox, 0.8 V')
    plt.legend(loc='upper right')
    plt.ylim([0, 1.2])
    plt.xlim([0,8])
    plt.xlabel('Thickness [um]')
    plt.savefig('20170201.pdf')
    plt.show()

    plt.figure(figsize=(10, 10))
    for i, voltage in enumerate([0.2, 0.4, 0.6, 0.8]):
        plot_series(dat, 'pedot', None, 2000, 'ox', voltage, color=colors10[i], label='%.1f V' % voltage)
    plt.title('Varied PEDOT and voltage, ox, 2000 rpm')
    plt.ylim([0, 1.2])
    plt.legend()
    plt.show()

    dat = read_kinetics(KineticsDataType.FinalL)
    print(dat)
    plt.figure(figsize=(10, 10))
    for i, pedot in enumerate([20, 30, 40, 60, 80]):
        plot_series(dat, 'voltage', pedot, 2000, None, None, color=colors10[i], label='%d perc PEDOT' % pedot)
    plt.title('Varied PEDOT and voltage, ox, 2000 rpm')
    plt.ylabel('Final L value')

    plt.legend(loc='lower right')
    plt.show()


main()
