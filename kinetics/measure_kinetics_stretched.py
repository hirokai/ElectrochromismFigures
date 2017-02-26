import os
import luigi
from common.util import chdir_root, ensure_folder_exists
from common.data_tools import load_csv
from common.figure_tools import colors10
from kinetics.measure_kinetics import CorrectedLValuesOfAllMovies
from common.luigi_tools import cleanup
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_stretched_kinetics():
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    plt.figure(figsize=(6, 4))

    in_path = os.path.join('data', 'stretched', 'corrected', '20161028', 'MVI_7962.csv')
    vs = load_csv(in_path, numpy=True)
    ts = np.array(range(len(vs)))
    plt.plot(ts, vs, c=colors10[0])
    plt.show()


def main():
    folders = {
        '20161028': os.path.join('/Volumes', 'ExtWork', 'Suda Electrochromism', '20161028')
    }
    for n, f in folders.iteritems():
        cleanup(CorrectedLValuesOfAllMovies(name=n, folder=f, mode='stretched'))
        luigi.run(
            ['CorrectedLValuesOfAllMovies', '--name', n, '--folder', f, '--mode', 'stretched', '--workers', '4',
             '--no-lock'])


def split_traces():
    folders = {
        '20161028': os.path.join('/Volumes', 'ExtWork', 'Suda Electrochromism', '20161028')
    }
    for n, f in folders.iteritems():
        cleanup(CorrectedLValuesOfAllMovies(name=n, folder=f, mode='stretched'))
        luigi.run(
            ['CorrectedLValuesOfAllMovies', '--name', n, '--folder', f, '--mode', 'stretched', '--workers', '4',
             '--no-lock'])


if __name__ == "__main__":
    chdir_root()
    main()
