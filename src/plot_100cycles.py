import luigi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

from data_tools import split_trace, save_csv, load_csv
from image_tools import do_cie_analysis
from luigi_tools import cleanup
from src.figure_tools import plot_and_save, set_common_format


def first_order(t, a_i, a_f, k, t0):
    y = a_i + (a_f - a_i) * (1 - np.exp(-k * (t - t0)))
    return y


def collect_all_cielab():
    root_folder = os.path.join(os.path.expanduser('~'), 'Google Drive/ExpDataLarge/Suda EC 100 cycles slices')
    roi = [906, 291, 40, 40]
    folder = os.path.join(root_folder, '100 cycles slices')
    labs = [do_cie_analysis(i, os.path.join(folder, name), roi) for i, name in enumerate(os.listdir(folder))]
    save_csv('../data/93-100cycles_new.csv', [['File number', 'L', 'a', 'b']] + labs)

    roi = [906, 291, 40, 40]
    folder = os.path.join(root_folder, '1-10 cycles slices')
    labs = [do_cie_analysis(i, os.path.join(folder, name), roi) for i, name in enumerate(os.listdir(folder))]
    save_csv('../data/1-9cycles_new.csv', [['File number', 'L', 'a', 'b']] + labs)


def get_l_vs_t(path1, path2):
    vs = np.array(load_csv(path1, skip_rows=1))
    offset = 62
    print(vs)
    ts1 = (vs[:, 0][offset::10].astype(float) - offset) / 60
    ls1 = vs[:, 1][offset::10].astype(float)

    vs = np.array(load_csv(path2, skip_rows=1))
    offset = 0
    ts2 = (vs[:, 0][offset::10].astype(float) - offset) / 60
    ls2 = vs[:, 1][offset::10].astype(float) * 1.15

    return [ts1, ls1, ts2, ls2]


def plot_l_vs_t(l_vs_t):
    def func():
        fig = plt.figure(figsize=(4.5, 3))
        ts1, ls1, ts2, ls2 = l_vs_t
        ax = fig.add_subplot(121)
        plt.xlim([0, 16])
        plt.ylim([0, 16])

        majorLocator = MultipleLocator(8)
        minorLocator = MultipleLocator(2)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)

        majorLocator = MultipleLocator(4)
        minorLocator = MultipleLocator(2)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_minor_locator(minorLocator)

        plt.plot(ts1, ls1, c='b', lw=1)

        ax = fig.add_subplot(122)
        plt.xlim([0, 16])
        plt.ylim([0, 16])

        majorLocator = MultipleLocator(8)
        minorLocator = MultipleLocator(2)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)

        majorLocator = MultipleLocator(4)
        minorLocator = MultipleLocator(2)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_minor_locator(minorLocator)

        plt.plot(ts2, ls2, c='b', lw=1)

    return func


def plot_split_traces(l_vs_t):
    ts1, ls1, ts2, ls2 = l_vs_t
    tss, lss = split_trace(ts1, ls1, range(0, 20, 1))
    for ts, ls in zip(tss, lss):
        if len(ts) > 0:
            plt.scatter(ts - min(ts), ls)
            t0 = 2
            popt, pconv = curve_fit(first_order, ts2, ls2, [ls2[0], ls2[-1], 1, t0])
            ai, af, k, t0 = popt
            plt.plot(ts2, first_order(ts2, *popt))


class CollectCIELab100Cycles(luigi.Task):
    def output(self):
        return [luigi.LocalTarget('../data/1-9cycles_new.csv'), luigi.LocalTarget('../data/93-100cycles_new.csv')]

    def requires(self):
        return None

    def run(self):
        os.chdir(os.path.dirname(__file__))
        collect_all_cielab()


class Plot100Cycles(luigi.Task):
    name = luigi.Parameter()
    resources = {"matplotlib": 1}

    def requires(self):
        return CollectCIELab100Cycles()

    def output(self):
        return [luigi.LocalTarget('../dist/Fig ' + self.name + '.pdf')]

    def run(self):
        l_vs_t = get_l_vs_t(self.input()[0].path, self.input()[1].path)
        set_common_format()
        plot_and_save(plot_l_vs_t(l_vs_t), self.name)
        # plot_split_traces(l_vs_t)


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    cleanup(Plot100Cycles(name='3c'))
    luigi.run(['Plot100Cycles', '--name', '3c'])
