import matplotlib.pyplot as plt
import numpy as np
import luigi
from figure_tools import plot_and_save, set_common_format
from luigi_tools import cleanup


def plot_thickness_pedot():
    str = """80	348.54	41.3672
    60	795.32	20.09073
    40	2604.8	41.33207
    30	3121.2	39.43653
    20	3740.4	107.4703"""

    dat = np.array(map(lambda l: map(float, l.split('\t')), str.split('\n')))

    plt.figure(figsize=(4.5, 3))
    (_, caps, _) = plt.errorbar(dat[:, 0], dat[:, 1] / 1000, dat[:, 2] / 1000, lw=1, elinewidth=1)
    for cap in caps:
        cap.set_markeredgewidth(1)
    plt.scatter(dat[:, 0], dat[:, 1] / 1000, lw=0, s=10)
    plt.xlabel('PEDOT ratio [%]')
    plt.ylabel('Film thickness [um]')
    plt.axis([0, 100, 0, 6])


def plot_thickness_rpm():
    str = """500	5619	95.72617197
750	4368	72.11033213
1000	3832.2	39.49227773
2000	2856.2	37.55848772
3000	2627.2	64.63930693
4000	1831.2	51.92147918
5000	2199.4	57.51921418"""

    dat = np.array(map(lambda l: map(float, l.split('\t')), str.split('\n')))

    w = 8
    plt.figure(figsize=(4.5, 3))
    (_, caps, _) = plt.errorbar(dat[:, 0], dat[:, 1] / 1000, dat[:, 2] / 1000, lw=1, elinewidth=1)
    for cap in caps:
        cap.set_markeredgewidth(1)
    plt.scatter(dat[:, 0], dat[:, 1] / 1000, lw=0, s=10)
    plt.xlabel('Spin coating speed [rpm]')
    plt.ylabel('Film thickness [um]')
    plt.axis([0, 6000, 0, 6])


class PlotThickness(luigi.Task):
    name1 = luigi.Parameter()
    name2 = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget('../dist/Fig ' + self.name1 + '.pdf'),
                luigi.LocalTarget('../dist/Fig ' + self.name2 + '.pdf')]

    def run(self):
        set_common_format()
        plot_and_save(plot_thickness_pedot,self.name1)
        plot_and_save(plot_thickness_rpm,self.name2)


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    cleanup(PlotThickness(name1='S1', name2='S2'))
    luigi.run(['PlotThickness', 'name1', 'S1', 'name2', 'S2'])
