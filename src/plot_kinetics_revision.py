import os
import numpy as np
import matplotlib.pyplot as plt
from data_tools import colors10


class Kinetics:
    def __init__(self):
        self.dat = {}

    def __repr__(self):
        ks = self.dat.keys()
        return '%d entries: %s' % (len(ks), ' '.join(ks))

    @staticmethod
    def mk_key(pedot, rpm, mode, voltage):
        return '%d,%d,%s,%.1f' % (pedot, rpm, mode, voltage)

    @staticmethod
    def get_cond_from_key(k):
        p, r, m, v = k.split(',')
        return int(p), int(r), m, float(v)

    def set_data(self, pedot, rpm, mode, voltage, v):
        assert (pedot in [20, 30, 40, 60, 80])
        assert (rpm in [500, 1000, 2000, 3000, 4000, 5000])
        assert (mode in ['const', 'ox', 'red'])
        if mode == 'ox':
            assert voltage in [0, 0.2, 0.4, 0.6, 0.8]
        self.dat[self.mk_key(pedot, rpm, mode, voltage)] = v

    def get_data(self, pedot, rpm, mode, voltage):
        return self.dat.get(self.mk_key(pedot, rpm, mode, voltage))


def plot_series(dat, variable, pedot, rpm, mode, voltage, color=colors10[0], label=None, show=False):
    assert isinstance(dat, Kinetics)
    if variable == 'pedot':
        ps = [20, 30, 40, 60, 80]
        d = {p: dat.get_data(p, rpm, mode, voltage) for p in ps}
    elif variable == 'rpm':
        rs = [500, 1000, 2000, 3000, 4000, 5000]
        d = {r: dat.get_data(pedot, r, mode, voltage) for r in rs}
    elif variable == 'mode':
        ms = ['const', 'ox', 'red']
        d = {m: dat.get_data(pedot, rpm, m, voltage) for m in ms}
    elif variable == 'voltage':
        vs = [-0.5, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
        d = {v: dat.get_data(pedot, rpm, mode, v) for v in vs}
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


def read_rate_constant():
    dat = Kinetics()
    for rpm in [500, 1000, 2000, 3000, 4000, 5000]:
        for pedot in [20, 30, 40, 60, 80]:
            for voltage in [0.2, 0.4, 0.6, 0.8]:
                for date in ['20161013', '20161019']:
                    path = os.path.join('data', 'kinetics', 'fitted_manual', date,
                                        '%d perc PEDOT - %d rpm' % (pedot, rpm), 'ox %.1f.csv' % voltage)
                    if os.path.exists(path):
                        with open(path) as f:
                            t0, kinv, li, lf = map(float, f.read().strip().split(','))
                            k = 1.0 / kinv
                            d = dat.get_data(pedot, rpm, 'ox', voltage)
                            if d:
                                dat.set_data(pedot, rpm, 'ox', voltage, d + [k])
                            else:
                                dat.set_data(pedot, rpm, 'ox', voltage, [k])
    return dat


def read_final_l():
    dat = Kinetics()

    def read_mode(mode, voltages):
        for voltage in voltages:
            for date in ['20161013', '20161019']:
                path = os.path.join('data', 'kinetics', 'fitted_manual', date,
                                    '%d perc PEDOT - %d rpm' % (pedot, rpm), '%s %.1f.csv' % (mode, voltage))
                if os.path.exists(path):
                    with open(path) as f:
                        t0, kinv, li, lf = map(float, f.read().strip().split(','))
                        d = dat.get_data(pedot, rpm, mode, voltage)
                        if d:
                            dat.set_data(pedot, rpm, mode, voltage, d + [lf])
                        else:
                            dat.set_data(pedot, rpm, mode, voltage, [lf])

    for rpm in [500, 1000, 2000, 3000, 4000, 5000]:
        for pedot in [20, 30, 40, 60, 80]:
            read_mode('ox', [0, 0.2, 0.4, 0.6, 0.8])
            read_mode('red', [-0.5, -0.2])
    return dat


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))

    dat = read_rate_constant()
    plt.figure(figsize=(10, 10))
    for i, pedot in enumerate([20, 30, 40, 60, 80]):
        plot_series(dat,
                    'rpm', pedot, None, 'ox', 0.8, color=colors10[i], label='%d perc PEDOT' % pedot)
    plt.title('Varied PEDOT and rpm, ox, 0.8 V')
    plt.legend(loc='upper right')
    plt.ylim([0, 1.2])
    plt.close()

    plt.figure(figsize=(10, 10))
    for i, voltage in enumerate([0.2, 0.4, 0.6, 0.8]):
        plot_series(dat, 'pedot', None, 2000, 'ox', voltage, color=colors10[i], label='%.1f V' % voltage)
    plt.title('Varied PEDOT and voltage, ox, 2000 rpm')
    plt.ylim([0, 1.2])
    plt.legend()
    plt.close()

    dat = read_final_l()
    print(dat)
    plt.figure(figsize=(10, 10))
    for i, pedot in enumerate([20, 30, 40, 60, 80]):
        plot_series(dat, 'voltage', pedot, 2000, 'ox', None, color=colors10[i], label='%d perc PEDOT' % pedot)
    for i, pedot in enumerate([20, 30, 40, 60, 80]):
        plot_series(dat, 'voltage', pedot, 2000, 'red', None, color=colors10[i], label='%d perc PEDOT' % pedot)
    plt.title('Varied PEDOT and voltage, ox, 2000 rpm')
    plt.ylabel('Final L value')

    plt.legend(loc='lower right')
    plt.show()


main()
