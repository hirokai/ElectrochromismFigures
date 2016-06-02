import luigi
import csv
import matplotlib.pyplot as plt
import numpy as np
from data_tools import ambiguous_path, colors10
from measure_cie_space import CollectCIESpace
from figure_tools import figure
from luigi_tools import cleanup
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colors


@figure('1c')
def plot_cv():
    for i, rate in enumerate([5, 10, 50, 80, 100, 200]):
        path = os.path.join('../../Suda Electrochromism data/20160513 CV/', '70wt%% CV mem %dmV.TXT' % rate)
        print(path)
        vs2 = []
        with open(path, 'rU') as f:
            s = ''
            while s.rstrip() != 'Potential/V\tCurrent/A':
                s = f.readline()
            f.readline()  # Skip a blank line
            for line in f:
                ws = line.split('\t')
                if len(ws) == 2:
                    vs2.append(map(float, ws))
        arr = np.array(vs2)
        arr[:, 1] = arr[:, 1] * 1e6
        idxs = np.argwhere(arr[:, 0] == 0)
        idx1 = idxs[-3][0]
        idx2 = idxs[-1][0]
        v = [0.2, 0.4, 0.5, 0.6, 0.8, 1][i]
        plt.plot(arr[idx1:idx2, 0], arr[idx1:idx2, 1], c=colors.hsv_to_rgb((1, 1, v)))
    plt.xlim([-0.8, 1])
    plt.ylim([-600, 600])


class PlotCV(luigi.Task):
    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget('../Fig 1c.pdf')]

    def run(self):
        plot_cv()


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    cleanup(PlotCV())
    luigi.run(['PlotCV'])
