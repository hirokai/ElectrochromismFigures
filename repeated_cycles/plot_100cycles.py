import re
import os
import csv
import luigi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from common.data_tools import split_trace, save_csv, load_csv
from common.figure_tools import plot_and_save, set_common_format
from common.luigi_tools import cleanup
from kinetics.measure_kinetics import MakeAllSlices, RawLValuesOfSingleMovie, read_rois_simple
from common.util import bcolors, basename_noext, chdir_root, ensure_folder_exists
from measure_100cycles import MeasureAll100Cycles, Measure100Cycles


def plot_section(in_path):
    vs = load_csv(in_path, numpy=True)
    plt.plot(range(len(vs)), vs)
    plt.xlim([0, 1200])
    plt.ylim([0, 60])


class PlotAll100Cycles(luigi.Task):
    resources = {"matplotlib": 1}

    def requires(self):
        with open(os.path.join('parameters', 'dataset', '100cycles.csv'), 'rU') as f:
            reader = csv.DictReader(f)
            inputs = {row['name']: Measure100Cycles(name=row['name'], date=row['date'], folder=row['folder'],
                                                    initial=row['initial'], final=row['final']) for row in reader}
        return inputs

    def output(self):
        # print('PlotAll100Cycles.output()', list(self.input().iteritems()))
        return {name: luigi.LocalTarget(os.path.join('repeated_cycles', 'dist', '%s.pdf' % name)) for name, input in
                self.input().iteritems()}

    def run(self):
        for name, output in self.output().iteritems():
            sections = self.input()[name]
            set_common_format()
            plt.subplot(121)
            plot_section(sections[0].path)
            plt.subplot(122)
            plot_section(sections[1].path)

            ensure_folder_exists(output.path)
            plt.savefig(output.path)
            plt.clf()


def main():
    chdir_root()
    cleanup(PlotAll100Cycles())
    luigi.run(['PlotAll100Cycles'])


if __name__ == "__main__":
    main()
