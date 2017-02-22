import os
import csv

import luigi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from common.data_tools import ambiguous_path
from common.luigi_tools import cleanup
from kinetics.measure_cie_space import CollectCIESpace
from common.figure_tools import plot_and_save, set_common_format, colors10


def normalize(vs, idx):
    mn = min(vs)
    mx = vs[idx]
    return (vs - mn) / (mx - mn)


def plot_cie_space(ax=None):
    if ax is None:
        fig = plt.figure(figsize=(4.5, 3))
        ax = fig.axes()
    with open(ambiguous_path(os.path.join('data', 'cielab_space', '*.csv'))) as f:
        reader = csv.reader(f)
        vss = []
        for row in reader:
            vss.append(map(float, [row[0]] + row[2:]))

    vss = np.array(vss)
    vss[:, 0] = np.arange(0, 30, 0.1) - 2.25
    # Index for 15 sec
    idx_normalize = np.argmin(abs(vss[:, 0] - 15))

    vss[:, 1] = normalize(vss[:, 1], idx_normalize)
    vss[:, 2] = normalize(vss[:, 2], idx_normalize)
    vss[:, 3] = normalize(vss[:, 3], idx_normalize)
    # vss[:, 1] = (vss[:, 1] - 18) / (40 - 18)
    # vss[:, 2] = (vss[:, 2] - 18.5) / (38 - 18.5)
    # vss[:, 3] = (vss[:, 3] - 26) / (41.5 - 26)

    d = {'lw': 1} if ax is None else {}
    plt.plot(vss[:, 0], vss[:, 1], c=colors10[0], **d)
    plt.plot(vss[:, 0], vss[:, 2], c=colors10[1], **d)
    plt.plot(vss[:, 0], vss[:, 3], c=colors10[2], **d)
    plt.axis([0, 15, -0.1, 1.1])
    majorLocator = MultipleLocator(6)
    minorLocator = MultipleLocator(2)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.xlabel('Time [sec]')
    plt.ylabel('Normalized L* value')


class PlotCIESpace(luigi.Task):
    name = luigi.Parameter()

    def requires(self):
        return [CollectCIESpace(movie_name='04 MVI_0785 10fps')]

    def output(self):
        return [luigi.LocalTarget('../dist/Fig ' + self.name + '.pdf')]

    def run(self):
        set_common_format()
        plot_and_save(plot_cie_space, self.name)


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    cleanup(PlotCIESpace(name='4d'))
    luigi.run(['PlotCIESpace', '--name', '4d'])
