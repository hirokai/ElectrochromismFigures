import csv
import itertools
import os
import shutil
import sys
import unittest

import luigi
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from data_tools import colors10, load_csv, save_csv
from util import ensure_exists, ensure_folder_exists, basename_noext, bcolors

from kinetics.split import SplitTraces, split_all_traces, save_split_data
from make_slices import mk_slices
from measure_colorchart import MeasureLValuesOfColorCharts
from image_tools import get_cie_l_rois


#
# Handling movies, conditions, ROIs
#

def mk_condition_name(c):
    return '%d wt%% PEDOT, %d rpm' % (c['pedot'], c['rpm'])


def read_rois(path):
    with open(path) as f:
        reader = csv.reader(f)
        _ = reader.next()  # Skip first row
        obj = {}
        current_num = None
        for row in reader:
            if len(filter(lambda s: s != '', row)) <= 2:
                current_num = row[0]
            else:
                if current_num not in obj:
                    obj[current_num] = []
                obj[current_num].append(map(int, row[7:11]))
    return obj


#
# Making slices
#

# For testing only.
# Actual measurement invokes MakeSingleMovieSlices instead.
class MakeAllSlices(luigi.WrapperTask):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        print('MakeAllSlices: %s' % self.folder)
        movie_files = sorted(filter(os.path.isfile, [os.path.join(self.folder, n) for n in os.listdir(self.folder)]))
        for num, f in enumerate(movie_files):
            yield MakeSingleMovieSlices(path=f)


class MakeSingleMovieSlices(luigi.Task):
    path = luigi.Parameter()

    def run(self):
        assert (os.path.isfile(self.path)), 'Not a file: %s' % self.path
        print(bcolors.OKGREEN + 'MakeSingleMovieSlices: \nInput %s' % self.path + bcolors.ENDC)
        print(bcolors.OKGREEN + 'Output: %s' % self.output().path + bcolors.ENDC)
        mk_slices(self.path)

    def output(self):
        return luigi.LocalTarget(os.path.join(os.path.dirname(self.path), 'slices', basename_noext(self.path)))


#
# Measurement of slices
#

def measure_movie_slices(folder_path, roi_samples, max_timepoints=1000, rois_calibration=None):
    print(bcolors.OKGREEN + 'Measuring slices: ' + folder_path + bcolors.ENDC)
    files = sorted(os.listdir(folder_path))
    lss = np.zeros((3, max_timepoints))
    for i, name in enumerate(files):
        path = os.path.join(folder_path, name)
        if i % 10 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        ls = get_cie_l_rois(path, roi_samples)
        lss[:, i] = ls
    sys.stdout.write('\n')
    return np.array(lss)


def all_measure_cielab(folder, roi_path, out_path, max_timepoints=1000):
    folder_slices = os.path.join(folder, 'slices')
    folders = sorted(filter(lambda n: os.path.isdir(os.path.join(folder_slices, n)), os.listdir(folder_slices)))
    rois = read_rois(roi_path)
    print(sorted(rois.keys()), folders)
    print(rois)

    def func(f):
        num = f[4:8]
        fp = os.path.join(folder_slices, f)
        print('%s:%s' % (num, fp))
        lss = measure_movie_slices(fp, rois.get(num), max_timepoints=max_timepoints)
        return lss

    lsss = map(func, folders)

    result = np.array(lsss).transpose()
    print(result.shape)
    result = np.reshape(result, (max_timepoints, len(folders) * 3))
    print(result.shape)
    out_folder = os.path.dirname(out_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    np.savetxt(out_path, result, delimiter=",")


class RawLValuesOfSingleMovie(luigi.Task):
    name = luigi.Parameter()
    path = luigi.Parameter()
    roi = luigi.Parameter()

    def requires(self):
        return MakeSingleMovieSlices(path=self.path)

    def run(self):
        print('%s' % self.path)
        folder_path = os.path.join(os.path.dirname(self.path), 'slices', basename_noext(self.path))
        print(self.roi)
        if self.roi == '':
            print "No ROI"
        else:
            rois = list(grouper(4, [int(s) for s in str(self.roi).split(',')]))
            lss = measure_movie_slices(folder_path, rois)
            ensure_folder_exists(self.output().path)
            np.savetxt(self.output().path, lss.transpose(), delimiter=",")
            # with open(self.output().path) as f:
            #     writer = csv.writer(f)
            #     writer.writerow([for ls in lss])

    def output(self):
        return luigi.LocalTarget(os.path.join('data', 'kinetics', 'raw',
                                              self.name, "%s.csv" % basename_noext(self.path)))


# http://stackoverflow.com/questions/4998427/how-to-group-elements-in-python-by-n-elements
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


def flatten(vs):
    return reduce(lambda a, b: a + b, vs)


class RawLValuesOfAllMovies(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        def test(p):
            return os.path.isfile(p) and p[-4:].lower() == '.mov'

        movie_files = sorted(filter(test, [os.path.join(self.folder, n) for n in os.listdir(self.folder)]))
        # print(movie_files)
        roi_path = 'parameters/%s/rois.csv' % self.name
        rois = read_rois(roi_path)

        def mk(f):
            r = rois.get(os.path.basename(f)[4:8])
            if r is not None:
                return ','.join(map(str, flatten(r)))
            else:
                print('ROI not found: %s, %s, %s' % (os.path.basename(f)[4:8], f, ' '.join(rois.keys())))
                return ''

        return [RawLValuesOfSingleMovie(name=self.name, path=f, roi=mk(f))
                for f in movie_files]

    def output(self):
        return self.input()


#
# Correction by color chart
#

def correct_cielab(in_csvs, correction_csvs, out_csvs):
    assert all([isinstance(a, luigi.LocalTarget) for a in in_csvs])
    assert all([os.path.isfile(a) for a in correction_csvs])
    assert all([isinstance(a, luigi.LocalTarget) for a in out_csvs])

    for f1, p2, f3 in zip(in_csvs, correction_csvs, out_csvs):
        # print(f1.path,p2,f3.path)
        factor = load_csv(p2, numpy=True)[:, 1]
        raw = load_csv(f1.path, numpy=True)[0:len(factor), :]
        corrected = raw.transpose() * np.repeat([factor], 3, axis=0)
        ensure_folder_exists(f3.path)
        save_csv(f3.path, np.transpose(corrected))


class CorrectedLValuesOfAllMovies(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return [RawLValuesOfAllMovies(name=self.name, folder=self.folder),
                MeasureLValuesOfColorCharts(name=self.name,
                                            folder=self.folder,
                                            roipath=os.path.join('parameters', self.name, 'calibration_rois.csv'))
                ]

    def output(self):
        names = os.listdir(os.path.join('data', 'kinetics', 'raw', self.name))
        return [luigi.LocalTarget(
            os.path.join('data', 'kinetics', 'corrected', self.name, n)) for n
                in names]

    def run(self):
        base_folder = os.path.join('data', 'kinetics', 'correction', self.name)
        correction_value_csvs = [os.path.join(base_folder, n) for n in os.listdir(base_folder) if
                                 n.find('_1.csv') != -1]
        correct_cielab(self.input()[0], correction_value_csvs, self.output())


#
# Splitting traces
#

class SplitAllTraces(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return CorrectedLValuesOfAllMovies(name=self.name, folder=self.folder)

    def run(self):
        movie_conditions_csv = os.path.join('parameters', self.name, 'movie_conditions.csv')
        sample_conditions_csv = os.path.join('parameters', self.name, 'sample_conditions.csv')
        split_dat = split_all_traces([f.path for f in self.input()], movie_conditions_csv, sample_conditions_csv)
        assert isinstance(split_dat, SplitTraces)
        save_split_data(split_dat, self.name, self.output().path)

    def output(self):
        return luigi.LocalTarget(os.path.join('data', 'kinetics', 'split', '%s alldata.p' % self.name))


def remove_trailing_zeros(vs):
    idx = np.where(vs == 0)[0][0]
    # print('remove_trailing_zeros',idx)
    return vs[0:idx]


#
# First-order fitting.
#


def first_order(t, a_i, a_f, k, t0):
    y = a_i + (a_f - a_i) * (1 - np.exp(-k * (t - t0)))
    return y


def first_order_af_fixed(a_f):
    def func(t, a_i, k, t0):
        y = a_i + (a_f - a_i) * (1 - np.exp(-k * (t - t0)))
        return y

    return func


def read_manual_fitting(date, pedot, rpm, mode, voltage):
    path = os.path.join('data', 'kinetics', 'fitted_manual', date,
                        '%d perc PEDOT - %d rpm' % (pedot, rpm), '%s %.1f.csv' % (mode, voltage))
    if os.path.exists(path):
        with open(path) as f:
            t0, kinv, li, lf = map(float, f.read().strip().split(','))
        return [li, lf, 1.0 / kinv, t0]
    else:
        return None


# Using initial guess that is entered manually.
def plot_fitting_curve(ts, ys, c, p_initial):
    fit_start = 0
    fit_to = min(45, len(ts) - 1)
    if len(ts) <= max(fit_start, 10):
        return None
    try:
        t0, k, ai, af = p_initial
        popt2, pcov = curve_fit(first_order_af_fixed(af), ts[fit_start:fit_to], ys[fit_start:fit_to],
                                [ai, k, t0], method='lm')
        perr = np.sqrt(np.diag(pcov))
        ai, k, t0 = popt2
        print(p_initial, [t0, k, ai, af])
        # ts_fit = np.linspace(fit_start, fit_to, 100)
        ts_fit_plot = np.linspace(fit_start, 45, 100)
        ys_fit_plot = first_order_af_fixed(af)(ts_fit_plot, *popt2)
        # ys_fit_plot2 = first_order(ts_fit_plot, *p_initial)
        try:
            plt.plot(ts_fit_plot, ys_fit_plot, c=c, lw=1)
        except:
            pass
        # plt.plot(ts_fit_plot, ys_fit_plot2, c=c, lw=1, ls='--')
        return [t0, k, ai, af]
    except RuntimeError as e:
        print('Fitting failed.')


#
# Plotting data
#


def all_plot(dat, conditions, save_folder=None):
    size = (14, 3)
    for i in range(30):
        name = mk_condition_name(conditions[i])
        plt.figure(figsize=size)
        for j, k in enumerate(['const', 'ox', 'red']):
            for ys in dat[i][k]:
                xs = np.array(range(len(ys))) + (j * 800)
                # print(i, j, len(xs))
                plt.plot(xs, ys, c=colors10[0])
        plt.ylim([0, 60])
        plt.title('Sample #%d: %s' % (i + 1, name))
        if save_folder:
            plt.savefig(os.path.join(save_folder, '%s.png' % name))
        else:
            plt.show()


def plot_split_traces(dat, sample_conditions, save_folder=None):
    assert isinstance(dat, SplitTraces)
    rpms = [500, 1000, 2000, 3000, 4000, 5000]
    pedots = [20, 30, 40]
    modes = ['const', 'ox', 'red']
    voltage_dict = {
        'const': [0.8, -0.5],
        'ox': [0, 0.2, 0.4, 0.6, 0.8],
        # Corrected on 12/7, 2016.
        # See 20160512 Suda EC amperometry/analysis/voltage_profile.jl
        'red': [0.4, 0.2, 0, -0.2, -0.5]
    }
    for k, mode in enumerate(modes):
        plt.figure(figsize=(20, 15))
        for i, rpm in enumerate(rpms):
            for j, pedot in enumerate(pedots):
                plt.subplot(len(rpms), len(pedots), i * len(pedots) + j + 1)
                voltages = voltage_dict[mode]
                count = 0
                ls_max = 0
                ls_min = 100
                for voltage in voltages:
                    ts, ls = dat.get_data(pedot, rpm, mode, voltage)
                    if len(ts) > 0:
                        ls_max = max(ls_max, max(ls))
                        ls_min = min(ls_min, min(ls))
                        plt.scatter(ts - min(ts), ls, s=10, linewidth=0, c=colors10[count % 10])
                        # p_initial = read_manual_fitting("20161013", pedot, rpm, mode, voltage)
                        # if p_initial:
                        #     print(p_initial)
                        #     plot_fitting_curve(ts, ls, colors10[count % 10], p_initial)
                    count += 1
                plt.ylim([ls_min, ls_max])
                plt.title('%s: %d%%, %d rpm' % (mode, pedot, rpm))
        plt.show()
    return


def plot_single_split_trace(dataset_name, pedot, rpm, mode, voltage, accumulate=False):
    path = os.path.join('data', 'kinetics', 'split', dataset_name, '%d perc PEDOT - %d rpm' % (pedot, rpm),
                        '%s %.1f.csv' % (mode, voltage))
    print('Plotting: %s' % path)
    with open(path) as f:
        reader = csv.reader(f)
        vs = np.array([map(float, r) for r in reader])
    plt.plot(vs[:, 0], vs[:, 1])
    if not accumulate:
        plt.show()


class PlotKineticsData(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return SplitAllTraces(name=self.name, folder=self.folder)

    def run(self):
        conditions_csv = os.path.join('parameters', self.name, 'sample_conditions.csv')
        conditions = read_sample_conditions(conditions_csv)
        split_dat = read_all_split_traces('data/kinetics_split/alldata.p')
        save_folder = os.path.join('dist', 'kinetics_revision', self.name)
        ensure_exists(save_folder)
        plot_split_traces(split_dat, conditions, save_folder=save_folder)
        all_plot(split_dat, conditions, save_folder=save_folder)

    def output(self):
        return luigi.LocalTarget('dist/kinetics_revision')


class PlotSingleKineticsData(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return SplitAllTraces(name=self.name, folder=self.folder)

    def run(self):
        # split_dat = read_all_split_traces('data/kinetics_split/%s alldata.p' % self.name)
        for pedot in [20, 30, 40]:
            for rpm in [500, 1000, 2000]:
                plot_single_split_trace(self.name, pedot, rpm, 'ox', 0.8, accumulate=True)
        plt.show()


class PlotAllTraces(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return SplitAllTraces(name=self.name, folder=self.folder)

    def run(self):
        conditions_csv = os.path.join('parameters', self.name, 'sample_conditions.csv')
        conditions = read_sample_conditions(conditions_csv)
        split_dat = read_all_split_traces(os.path.join('data', 'kinetics', 'split', '%s alldata.p' % self.name))
        save_folder = os.path.join('dist', 'kinetics_revision', self.name)
        ensure_exists(save_folder)
        plot_split_traces(split_dat, conditions, save_folder=save_folder)
        # all_plot(split_dat, conditions, save_folder=save_folder)

    def output(self):
        return luigi.LocalTarget('dist/kinetics_revision')


#
# Testing
#


class TestKineticsAll(unittest.TestCase):
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))

    def test_MakeAllSlices(self):
        r = luigi.run(['MakeAllSlices', '--name', '20161013', '--folder',
                       '/Volumes/ExtWork/Suda Electrochromism/20161013/', '--no-lock'])
        self.assertTrue(r)

    def test_RawLValuesOfSingleMovie(self):
        r = luigi.run(['RawLValuesOfSingleMovie', '--name', '20161013', '--path',
                       '/Volumes/ExtWork/Suda Electrochromism/20161013/MVI_7877.MOV',
                       '--roi', '100,100,20,20', '--no-lock'])
        self.assertTrue(r)

    def test_RawLValuesOfAllMovies(self):
        # shutil.rmtree('data/kinetics/raw/20161013', ignore_errors=True)
        # shutil.rmtree('data/kinetics/raw/20161013 all_l_values.csv', ignore_errors=True)
        r = luigi.run(['RawLValuesOfAllMovies', '--name', '20161019', '--folder',
                       '/Volumes/ExtWork/Suda Electrochromism/20161019/', '--workers', '4', '--no-lock'])
        self.assertTrue(r)

    def test_CorrectedLValuesOfAllMovies(self):
        # shutil.rmtree('data/kinetics/corrected', ignore_errors=True)
        r = luigi.run(['CorrectedLValuesOfAllMovies', '--name', '20161013', '--folder',
                       '/Volumes/ExtWork/Suda Electrochromism/20161013/', '--workers', '4', '--no-lock'])
        self.assertTrue(r)

    def test_SplitAllTraces(self):
        shutil.rmtree('data/kinetics/split', ignore_errors=True)
        r = luigi.run(['SplitAllTraces', '--name', '20161013', '--folder',
                       '/Volumes/ExtWork/Suda Electrochromism/20161013/', '--workers', '4', '--no-lock'])
        self.assertTrue(r)

    def test_SplitAllTraces2(self):
        shutil.rmtree('data/kinetics/split', ignore_errors=True)
        r = luigi.run(['SplitAllTraces', '--name', '20161019', '--folder',
                       '/Volumes/ExtWork/Suda Electrochromism/20161019/', '--workers', '4', '--no-lock'])
        self.assertTrue(r)

    # def test_PlotSingleKineticsData(self):
    #     shutil.rmtree('data/kinetics/split', ignore_errors=True)
    #     r = luigi.run(['PlotSingleKineticsData', '--name', '20161013', '--folder',
    #                    '/Volumes/ExtWork/Suda Electrochromism/20161013/', '--workers', '1', '--no-lock'])
    #     self.assertTrue(r)

    def test_PlotAllTraces(self):
        shutil.rmtree('dist/kinetics_revision', ignore_errors=True)
        r = luigi.run(['PlotAllTraces', '--name', '20161013', '--folder',
                       '/Volumes/ExtWork/Suda Electrochromism/20161013/', '--workers', '1', '--no-lock'])
        self.assertTrue(r)


#
# Main function
#


class MeasureAndPlotAll(luigi.WrapperTask):
    def requires(self):
        folders = {'20161013': '/Volumes/ExtWork/Suda Electrochromism/20161013/',
                   '20161019': '/Volumes/ExtWork/Suda Electrochromism/20161019/'}
        # folders_test = {'20161013': '/Volumes/ExtWork/Suda Electrochromism/20161013/'}
        tasks = [PlotAllTraces(name=k, folder=v) for k, v in folders.iteritems()]
        for t in tasks:
            yield t


def main():
    # logger = logging.getLogger('luigi-interface')
    # logger.setLevel(logging.WARNING)
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    # folders = {'20161013': '/Volumes/ExtWork/Suda Electrochromism/20161013/',
    #            '20161019': '/Volumes/ExtWork/Suda Electrochromism/20161019/'}
    if os.path.exists('data/kinetics_split/20161013 alldata.p'):
        os.remove('data/kinetics_split/20161013 alldata.p')
    out_folder = 'dist/kinetics_revision'
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    # luigi.run(['MeasureAndPlotAll'])
    luigi.run(['CorrectedLValues', '--name', '20161013', '--folder',
               '/Volumes/ExtWork/Suda Electrochromism/20161013/'])


if __name__ == "__main__":
    unittest.main()
