import luigi
import csv
import matplotlib.pyplot as plt
import numpy as np
from data_tools import ambiguous_path, colors10
from measure_cie_space import CollectCIESpace
from figure_tools import figure, set_common_format
from luigi_tools import cleanup


@figure('4d')
def plot_cie_space():
    plt.figure(figsize=(4.5,3))
    with open(ambiguous_path('../data/cielab_space/*.csv')) as f:
        reader = csv.reader(f)
        vss = []
        for row in reader:
            vss.append(map(float, [row[0]] + row[2:]))

    vss = np.array(vss)
    vss[:, 0] = np.arange(0, 30, 0.1) - 2.25

    vss[:, 1] = (vss[:, 1] - 18) / (40 - 18)
    vss[:, 2] = (vss[:, 2] - 18.5) / (38 - 18.5)
    vss[:, 3] = (vss[:, 3] - 26) / (41.5 - 26)

    plt.plot(vss[:, 0], vss[:, 1], c=colors10[0], lw=1)
    plt.plot(vss[:, 0], vss[:, 2], c=colors10[1], lw=1)
    plt.plot(vss[:, 0], vss[:, 3], c=colors10[2], lw=1)
    plt.axis([0, 15, -0.1, 1.1])


class PlotCIESpace(luigi.Task):
    def requires(self):
        return [CollectCIESpace(movie_name='04 MVI_0785 10fps')]

    def output(self):
        return [luigi.LocalTarget('../dist/Fig 4d.pdf')]

    def run(self):
        set_common_format()
        plot_cie_space()


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    cleanup(PlotCIESpace())
    luigi.run(['PlotCIESpace'])
