import os

import luigi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from src.data_tools import colors10
from src.luigi_tools import cleanup
from src.figure_tools import set_common_format, plot_and_save


def plot_cv():
    fig, ax = plt.subplots(figsize=(4.5, 3))
    path = os.path.join('../data/2000 rpm CV.txt')
    vs2 = []
    with open(path, 'rU') as f:
        s = ''
        while s.find('Potential/V') != 0:
            s = f.readline()
        f.readline()  # Skip a blank line
        for line in f:
            ws = line.split('\t')
            if len(ws) >= 2:
                vs2.append(map(float, ws))
    arr = np.array(vs2)
    arr[:, 1] = arr[:, 1] * 1e6
    idxs = np.argwhere(arr[:, 0] == 0)
    idx1 = idxs[-3][0]
    idx2 = idxs[-1][0]
    plt.plot(arr[idx1:idx2, 0], arr[idx1:idx2, 1], c='b', lw=1)
    plt.xlim([-0.8, 1])
    plt.ylim([-200, 200])
    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))


class PlotCV(luigi.Task):
    name = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget('../dist/Fig ' + self.name + '.pdf')]

    def run(self):
        set_common_format()
        plot_and_save(plot_cv, self.name)


def plot_cv2():
    fig, ax = plt.subplots(figsize=(4.5, 3))
    path = os.path.join('../data/cv/1021/74 30wt 2krpm.txt')
    vs2 = []
    with open(path, 'rU') as f:
        s = ''
        while s.find('Potential/V') != 0:
            s = f.readline()
        f.readline()  # Skip a blank line
        for line in f:
            ws = line.split(', ')
            if len(ws) >= 2:
                vs2.append(map(float, ws))
    arr = np.array(vs2)
    arr[:, 1] = arr[:, 1] * 1e6

    # Choose last cycle
    idxs = np.argwhere(arr[:, 0] == 0)
    # Potting part of split traces
    idx1 = idxs[-5][0]
    idx2 = idxs[-3][0]
    plt.plot(arr[idx1:idx2, 0], arr[idx1:idx2, 1], c=colors10[0], lw=0.5)
    idx1 = idxs[-3][0]
    idx2 = idxs[-1][0]
    plt.plot(arr[idx1:idx2, 0], arr[idx1:idx2, 1], c=colors10[1], lw=0.5)
    # idx1 = 0
    # idx2 = -1

    # Set range
    plt.xlim([-0.8, 1])
    plt.ylim([-200, 200])

    # Tick formatting
    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))


class PlotCV2(luigi.Task):
    name = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget('../dist/Fig ' + self.name + '.pdf')]

    def run(self):
        set_common_format()

        # def plot_cv_all():
        #     plot_cv2(only_last=False)
        #
        plot_and_save(plot_cv2, self.name)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    # cleanup(PlotCV(name='2b'))
    # luigi.run(['PlotCV', '--name', '2b'])
    cleanup(PlotCV2(name='cv_revision'))
    luigi.run(['PlotCV2', '--name', 'cv_revision'])
