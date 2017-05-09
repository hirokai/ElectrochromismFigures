import os
import csv
import luigi
from common.util import chdir_root, basename_noext
from common.data_tools import load_csv, mk_dict
from kinetics.measure_kinetics import RawLValuesOfAllMovies, CorrectedLValuesOfAllMovies
import shutil


def get_roi(dat, name):
    k = basename_noext(name)
    if k not in dat:
        k = k[4:8]
    return ','.join(dat[k])


class Measure1200Cycles(luigi.Task):
    name = luigi.Parameter()
    date = luigi.Parameter()
    folder = luigi.Parameter()

    # initial = luigi.Parameter()  # Early cycles
    # final = luigi.Parameter()  # Late cycles

    def requires(self):
        return CorrectedLValuesOfAllMovies(name=self.date, folder=self.folder, mode='100cycles')

    def run(self):
        # shutil.copy(os.path.join('data', '100cycles', 'corrected', self.date, basename_noext(self.initial) + '.csv'),
        #             self.output()['initial'].path)
        # shutil.copy(os.path.join('data', '100cycles', 'corrected', self.date, basename_noext(self.final) + '.csv'),
        #             self.output()['final'].path)
        pass

    def output(self):
        return {
            'initial': luigi.LocalTarget(os.path.join('data', '100cycles', 'corrected', '%s_initial.csv' % self.name)),
            'final': luigi.LocalTarget(os.path.join('data', '100cycles', 'corrected', '%s_final.csv' % self.name))
        }


def main():
    chdir_root()

    # For preparation of slices to define ROIs.
    # luigi.run(['MakeAllSlices', '--workers', '4',
    #            '--name', '20170502',
    #            '--folder', '/Volumes/ExtWork/Suda Electrochromism/20170502 More than 1000 cycles'])

    luigi.run(['Measure1200Cycles', '--workers', '4',
               '--name', '20170502', '--date', '20170502',
               '--folder', '/Volumes/ExtWork/Suda Electrochromism/20170502 More than 1000 cycles'])


if __name__ == "__main__":
    main()
