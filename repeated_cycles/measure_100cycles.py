import os
import csv
import luigi
from common.util import chdir_root, basename_noext
from common.data_tools import load_csv, mk_dict
from kinetics.measure_kinetics import RawLValuesOfSingleMovie
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
        dat = mk_dict(load_csv(os.path.join('parameters', self.date, 'sample rois.csv'))[1:])
        roi_initial = get_roi(dat, self.initial)
        roi_final = get_roi(dat, self.final)
        return [
            RawLValuesOfSingleMovie(name=self.name, path=os.path.join(self.folder, self.initial), roi=roi_initial,
                                    mode='100cycles'),
            RawLValuesOfSingleMovie(name=self.name, path=os.path.join(self.folder, self.final), roi=roi_final,
                                    mode='100cycles')
        ]

    def run(self):
        shutil.copy(self.input()[0].path, self.output()[0].path)
        shutil.copy(self.input()[1].path, self.output()[1].path)

    def output(self):
        return [
            luigi.LocalTarget(os.path.join('data', '100cycles', 'corrected', '%s_initial.csv' % self.name)),
            luigi.LocalTarget(os.path.join('data', '100cycles', 'corrected', '%s_final.csv' % self.name))
        ]


class MeasureAll100Cycles(luigi.WrapperTask):
    def requires(self):
        with open(os.path.join('parameters', 'dataset', '100cycles.csv'), 'rU') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield Measure100Cycles(name=row['name'], date=row['date'], folder=row['folder'],
                                       initial=row['initial'], final=row['final'])


def main():
    chdir_root()
    luigi.run(['MeasureAll100Cycles'])


if __name__ == "__main__":
    main()
