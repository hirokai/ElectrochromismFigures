import os
import csv
import luigi
from common.util import chdir_root, basename_noext
from common.data_tools import load_csv, mk_dict
from kinetics.measure_kinetics import RawLValuesOfSingleMovie, CorrectedLValuesOfAllMovies
import shutil


def get_roi(dat, name):
    k = basename_noext(name)
    if k not in dat:
        k = k[4:8]
    return ','.join(dat[k])


class Measure100Cycles(luigi.Task):
    name = luigi.Parameter()
    date = luigi.Parameter()
    folder = luigi.Parameter()
    initial = luigi.Parameter()  # Early cycles
    final = luigi.Parameter()  # Late cycles

    def requires(self):
        return CorrectedLValuesOfAllMovies(name=self.date, folder=self.folder, mode='100cycles')

    def run(self):
        shutil.copy(os.path.join('data', '100cycles', 'corrected', self.date, basename_noext(self.initial) + '.csv'),
                    self.output()['initial'].path)
        shutil.copy(os.path.join('data', '100cycles', 'corrected', self.date, basename_noext(self.final) + '.csv'),
                    self.output()['final'].path)

    def output(self):
        return {
            'initial': luigi.LocalTarget(os.path.join('data', '100cycles', 'corrected', '%s_initial.csv' % self.name)),
            'final': luigi.LocalTarget(os.path.join('data', '100cycles', 'corrected', '%s_final.csv' % self.name))
        }


class MeasureAll100Cycles(luigi.WrapperTask):
    def requires(self):
        with open(os.path.join('parameters', 'dataset', '100cycles.csv'), 'rU') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['name'].find('#') != 0:
                    yield Measure100Cycles(name=row['name'], date=row['date'], folder=row['folder'],
                                           initial=row['initial'], final=row['final'])


def main():
    chdir_root()
    luigi.run(['MeasureAll100Cycles', '--workers', '4'])


if __name__ == "__main__":
    main()
