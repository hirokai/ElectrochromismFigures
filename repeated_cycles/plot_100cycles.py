import re

import luigi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

from src.data_tools import split_trace, save_csv, load_csv
from src.figure_tools import plot_and_save, set_common_format
from src.image_tools import do_cie_analysis
from src.luigi_tools import cleanup
from src.measure_kinetics_revision import MakeAllSlices, RawLValuesOfSingleMovie, read_rois_simple
from src.util import bcolors, basename_noext


def first_order(t, a_i, a_f, k, t0):
    y = a_i + (a_f - a_i) * (1 - np.exp(-k * (t - t0)))
    return y


def collect_all_cielab(output_folder):
    print('collect_all_cielab(): %s' % output_folder)
    return
    root_folder = os.path.join(os.path.expanduser('~'), 'Google Drive/ExpDataLarge/Suda EC 100 cycles slices')
    roi = [906, 291, 40, 40]
    folder = os.path.join(root_folder, '100 cycles slices')
    labs = [do_cie_analysis(i, os.path.join(folder, name), roi) for i, name in enumerate(os.listdir(folder))]
    save_csv('../data/93-100cycles_new.csv', [['File number', 'L', 'a', 'b']] + labs)

    roi = [906, 291, 40, 40]
    folder = os.path.join(root_folder, '1-10 cycles slices')
    labs = [do_cie_analysis(i, os.path.join(folder, name), roi) for i, name in enumerate(os.listdir(folder))]
    save_csv('../data/1-9cycles_new.csv', [['File number', 'L', 'a', 'b']] + labs)


def get_l_vs_t(path1, path2, interval=1, time_omitted=False):
    print(bcolors.OKGREEN + 'Files reading: %s, %s' % (path1, path2) + bcolors.ENDC)
    vs = np.array(load_csv(path1, skip_rows=1))
    offset = 62
    print(vs)
    ls1 = vs[:, 1][offset::interval].astype(float)
    if time_omitted:
        ts1 = np.array(range(len(ls1))).astype(float) / 60
    else:
        ts1 = (vs[:, 0][offset::interval].astype(float) - offset) / 60

    vs = np.array(load_csv(path2, skip_rows=1))
    offset = 0
    ls2 = vs[:, 1][offset::interval].astype(float)
    if time_omitted:
        ts2 = np.array(range(len(ls2))).astype(float) / 60
    else:
        ts2 = (vs[:, 0][offset::interval].astype(float) - offset) / 60

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
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def output(self):
        return [
            luigi.LocalTarget(os.path.join('data', '100cycles', self.name, '1-9cycles.csv')),
            luigi.LocalTarget(os.path.join('data', '100cycles', self.name, '93-100cycles.csv')),
        ]

    def requires(self):
        return MakeAllSlices(name=self.name, folder=self.folder)

    def run(self):
        os.chdir(os.path.dirname(__file__))
        collect_all_cielab(self.output()[0].path)
        collect_all_cielab(self.output()[1].path)


class MovieWithROI():
    def __init__(self, movie_relpath, roi_path=None, movie_root=None):
        if movie_root is None:
            movie_root = os.path.join('/Volumes', 'ExtWork', 'Suda Electrochromism')
        self.movie_relpath = os.path.join(movie_root, movie_relpath)
        if roi_path is None:
            self.roi_path = os.path.join('parameters', movie_relpath.split(os.sep)[0], 'sample rois.csv')
        else:
            self.roi_path = roi_path

    def read_roi(self):
        n = os.path.basename(self.movie_relpath)[4:8]
        if n is None or not re.match(r"""\d+""", n):
            n = basename_noext(self.roi_path)
        obj = read_rois_simple(self.roi_path)
        print('MovieWithROI.read_roi(): ', obj)
        return ','.join(map(str, obj[n][0]))


class Plot100Cycles2(luigi.Task):
    name = luigi.Parameter()
    resources = {"matplotlib": 1}

    def requires(self):
        datasets = {'20161114': [MovieWithROI('20161114/MVI_7984.MOV'), MovieWithROI('20161114/MVI_7985.MOV')],
                    '20161115': [MovieWithROI('20161115/MVI_8020.MOV'), MovieWithROI('20161115/MVI_8021.MOV')]}
        obj = {}
        for k, v in datasets.iteritems():
            obj[k + '_0'] = RawLValuesOfSingleMovie(name=k, path=v[0].movie_relpath, roi=v[0].read_roi(),
                                                    mode='100cycles')
            obj[k + '_1'] = RawLValuesOfSingleMovie(name=k, path=v[1].movie_relpath, roi=v[1].read_roi(),
                                                    mode='100cycles')
        return obj

    def output(self):
        return [luigi.LocalTarget('./dist/Fig ' + self.name + '_a.pdf'),
                luigi.LocalTarget('./dist/Fig ' + self.name + '_b.pdf')]

    def run(self):
        set_common_format()
        names = ['20161115', '20161114']
        for name, suf in zip(names, ['a', 'b', 'c', 'd']):
            l_vs_t = get_l_vs_t(self.input()[name+'_0'].path, self.input()[name+'_1'].path, time_omitted=True)
            testing = False
            if testing:
                t1, l1, t2, l2 = l_vs_t
                plt.subplot(121)
                plt.plot(t1, l1)
                plt.ylim([0, 60])
                plt.subplot(122)
                plt.plot(t2, l2)
                plt.ylim([0, 60])
                plt.show()
            # なぜかこれでプロットが見えない。枠だけ見える。
            plot_and_save(plot_l_vs_t(l_vs_t), str(self.name) + "_" + suf)


if __name__ == "__main__":
    import os

    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    # cleanup(Plot100Cycles(name='3c'))
    # luigi.run(['Plot100Cycles', '--name', '3c'])

    cleanup(Plot100Cycles2(name='100cycles_revision'))
    luigi.run(['Plot100Cycles2', '--name', '100cycles_revision'])
