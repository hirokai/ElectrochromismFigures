import os
import luigi
from measure_kinetics_revision import all_make_slices, all_measure_cielab


class MakeAllSlices(luigi.Task):
    folder = luigi.Parameter()

    def run(self):
        all_make_slices(self.folder)

    def output(self):
        return [luigi.LocalTarget(os.path.join(self.folder, 'slices'))]


class RawLValues100Cycles(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return MakeAllSlices(folder=self.folder)

    def output(self):
        return luigi.LocalTarget(os.path.join('data', '100cycles', '%s all_l_values.csv' % self.name))

    def run(self):
        roi_path = 'parameters/%s/sample rois.csv' % self.name
        all_measure_cielab(self.folder, roi_path, self.output().path, max_timepoints=2000)


class MeasureAll100Cycles(luigi.WrapperTask):
    def requires(self):
        yield RawLValues100Cycles(name='20161024',folder='/Volumes/Mac Ext 2/Suda Electrochromism/20161024')
        yield RawLValues100Cycles(name='20151201', folder='/Volumes/Mac Ext 2/Suda Electrochromism/20161201')


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    luigi.run(
        ['MeasureAll100Cycles'])


if __name__ == "__main__":
    main()