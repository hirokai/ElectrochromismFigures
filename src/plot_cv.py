import luigi
from figure_tools import figure, set_common_format
from luigi_tools import cleanup
import matplotlib.pyplot as plt
import numpy as np
import os


@figure('2a', show=False)
def plot_cv():
    path = os.path.join('../data/2000 rpm CV.txt')
    print(path)
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
    print(vs2)
    arr = np.array(vs2)
    arr[:, 1] = arr[:, 1] * 1e6
    idxs = np.argwhere(arr[:, 0] == 0)
    idx1 = idxs[-3][0]
    idx2 = idxs[-1][0]
    plt.plot(arr[idx1:idx2, 0], arr[idx1:idx2, 1], c='b')
    plt.xlim([-0.8, 1])

    plt.ylim([-200, 200])


class PlotCV(luigi.Task):
    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget('../dist/Fig 2a.pdf')]

    def run(self):
        set_common_format()
        plot_cv()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    cleanup(PlotCV())
    luigi.run(['PlotCV'])
