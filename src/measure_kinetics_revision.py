import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from data_tools import colors10, split_trace, load_csv
from scipy.optimize import curve_fit
from util import ensure_exists, ensure_folder_exists, basename_noext, bcolors
from image_tools import get_cie_l_rois
import luigi
import shutil
import cPickle as pickle
from make_slices import mk_slices
import unittest
from luigi_tools import cleanup
import itertools


#
# Handling movies, conditions, ROIs
#

def mk_condition_name(c):
    return '%d wt%% PEDOT, %d rpm' % (c['pedot'], c['rpm'])


def read_sample_conditions(path):
    with open(path) as f:
        reader = csv.reader(f)
        reader.next()
        sample_conditions = [{'pedot': int(row[2]), 'rpm': int(row[1])} for row in reader]
    return sample_conditions


def read_movie_conditions(path):
    with open(path) as f:
        reader = csv.reader(f)
        reader.next()
        movie_conditions = [{'name': row[0], 'mode': row[1], 'samples': [int(s) for s in row[2].split(',')]} for row in
                            reader]
    return movie_conditions


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
        roi_path = 'parameters/%s/rois.csv' % self.name
        rois = read_rois(roi_path)
        # print(sorted(rois.keys()), folders)
        # print(rois)
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

def correct_cielab_stub(in_csvs, out_csvs):
    import shutil
    print(in_csvs, out_csvs)
    for f1, f2 in zip(in_csvs, out_csvs):
        ensure_folder_exists(f2.path)
        shutil.copyfile(f1.path, f2.path)


class CorrectedLValuesOfAllMovies(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return RawLValuesOfAllMovies(name=self.name, folder=self.folder)

    def output(self):
        names = os.listdir(os.path.join('data', 'kinetics', 'raw', self.name))
        return [luigi.LocalTarget(
            os.path.join('data', 'kinetics', 'corrected', self.name, n)) for n
                in names]

    def run(self):
        # FIXME: Stub
        correct_cielab_stub(self.input(),
                            self.output())


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


class SplitTraces:
    def __init__(self):
        self.dat = {}

    def __repr__(self):
        ks = self.dat.keys()
        return '%d entries: %s' % (len(ks), ' '.join(ks))

    @staticmethod
    def mk_key(pedot, rpm, mode, voltage):
        return '%d,%d,%s,%.1f' % (pedot, rpm, mode, voltage)

    @staticmethod
    def get_cond_from_key(k):
        p, r, m, v = k.split(',')
        return int(p), int(r), m, float(v)

    def set_data(self, pedot, rpm, mode, voltage, ts, vs):
        assert (pedot in [20, 30, 40, 60, 80])
        assert (rpm in [500, 1000, 2000, 3000, 4000, 5000])
        assert (mode in ['const', 'ox', 'red'])
        assert len(ts) == len(vs)
        if mode == 'ox':
            assert voltage in [0, 0.2, 0.4, 0.6, 0.8]
        self.dat[self.mk_key(pedot, rpm, mode, voltage)] = (ts, vs)

    def get_data(self, pedot, rpm, mode, voltage):
        return self.dat[self.mk_key(pedot, rpm, mode, voltage)]


def organize_data(dat_path, conditions_path):
    with open(dat_path) as f:
        reader = csv.reader(f)
        rows = np.array([map(float, r) for r in reader])
    dat = {}
    count = 0

    sample_conditions = read_sample_conditions(conditions_path)

    for i in range(rows.shape[1]):
        series = count / 3
        pos = count % 3
        # pos = 0: const, 1: oxd, 2: red
        if (series + 1) in [3, 7, 8, 9, 13, 23]:
            # print('swapped.',(series+1))
            pos = 2 - pos  # Irregular order.

        if series not in dat:
            dat[series] = {}
        # print(np.argmin(rows[:,i]))
        # row_truncated = rows[:, i][0:np.argmin(rows[:, i])]
        dat[series][pos] = rows[:, i]
        # print(series, pos)
        count += 1
    return dat, sample_conditions


def remove_trailing_zeros(vs):
    idx = np.where(vs == 0)[0][0]
    # print('remove_trailing_zeros',idx)
    return vs[0:idx]


def split_for_mode(res, ts, vs, mode, pedot, rpm):
    assert isinstance(res, SplitTraces)
    assert (pedot in [20, 30, 40, 60, 80])
    assert (rpm in [500, 1000, 2000, 3000, 4000, 5000])
    assert (mode in ['const', 'ox', 'red'])

    tss, vss = split_trace(ts, vs, range(2, 1000, 60))

    voltage_dict = {  # FIXME: Check values.
        'const': [0.8, -0.5, 0.8, -0.5, 0.8, -0.5],
        'ox': [0, 0.2, 0.4, 0.6, 0.8],
        'red': [0.1, -0.1, -0.3, -0.5, -0.7]
    }

    voltages = voltage_dict[mode]
    if mode == 'const':
        selected = zip(tss[1:], vss[1:])
    else:
        selected = zip(tss[3::2], vss[3::2])
    for i, ys in enumerate(selected):
        if i >= len(voltages):
            break
        ts = ys[0]
        vs = ys[1]
        voltage = voltages[i]
        res.set_data(pedot, rpm, mode, voltage, ts, vs)


def split_all_traces(in_csvs, movie_conditions_csv, sample_conditions_csv):
    movie_conditions = read_movie_conditions(movie_conditions_csv)
    sample_conditions = read_sample_conditions(sample_conditions_csv)
    res = SplitTraces()

    # Iterate through movie
    movie_num = 1
    for in_csv, movie_cond in zip(in_csvs, movie_conditions):
        vss = load_csv(in_csv, 0, numpy=True)
        assert vss.shape[1] == len(movie_cond['samples'])
        mode = movie_cond['mode']
        for sample_num, vs in zip(movie_cond['samples'], vss.T):
            vs2 = remove_trailing_zeros(vs)
            cond = sample_conditions[sample_num - 1]
            pedot = cond['pedot']
            rpm = cond['rpm']
            # Triple of (mode, pedot, rpm) uniquely identifies the sample in the set of movies.
            t = range(len(vs2))
            split_for_mode(res, t, vs2, mode, pedot, rpm)
        movie_num += 1
    return res


def save_split_data(dat, dataset_name, p_path):
    assert isinstance(dat, SplitTraces)

    # Save as separate csv files.
    for k, d in dat.dat.iteritems():
        pedot, rpm, mode, voltage = SplitTraces.get_cond_from_key(k)
        out_path = os.path.join('data', 'kinetics', 'split', dataset_name,
                                '%d perc PEDOT - %d rpm' % (pedot, rpm),
                                '%s %.1f.csv' % (mode, voltage))
        ts, vs = d
        ensure_folder_exists(out_path)
        rows = map(list, zip(*[[str(t) for t in ts], [str(v) % v for v in vs]]))
        with open(out_path, 'wb') as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)

    # Also, save as a picked data.
    ensure_folder_exists(p_path)
    with open(p_path, 'wb') as f:
        pickle.dump(dat, f)


def read_all_split_traces(path):
    with open(path) as f:
        obj = pickle.load(f)
    return obj


#
# First-oder fitting.
#


def first_order(t, a_i, a_f, k, t0):
    y = a_i + (a_f - a_i) * (1 - np.exp(-k * (t - t0)))
    return y


def plot_fitting_curve(ts, ys, c):
    fit_start = 0
    fit_to = min(45, len(ts) - 1)
    if len(ts) <= max(fit_start, 10):
        return None
    try:
        for k_initial in [0.1, 0.3, 0.5]:
            popt, pcov = curve_fit(first_order, ts[fit_start:fit_to], ys[fit_start:fit_to],
                                   [ys[fit_start], ys[-1], k_initial, 2], method='lm')
            p_initial = [ys[fit_start], ys[fit_to], k_initial, 2]
            perr = np.sqrt(np.diag(pcov))
            print(perr)
            if not np.isinf(perr[0]):
                # ts_fit = np.linspace(fit_start, fit_to, 100)
                ts_fit_plot = np.linspace(fit_start, 45, 100)
                ys_fit_plot = first_order(ts_fit_plot, *popt)
                ys_fit_plot2 = first_order(ts_fit_plot, *p_initial)
                plt.plot(ts_fit_plot, ys_fit_plot, c=c, lw=1)
                plt.plot(ts_fit_plot, ys_fit_plot2, c=c, lw=1, ls='--')
        return popt
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
    voltage_dict = {  # FIXME: Check values.
        'const': [0.8, -0.5],
        'ox': [0, 0.2, 0.4, 0.6, 0.8],
        'red': [0.1, -0.1, -0.3, -0.5, -0.7]
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
                    # if mode == 'red':
                    #     ls = np.log(ls-min(ls)+0.01)
                    # else:
                    #     ls = np.log(max(ls) - ls + 0.01)
                    ls_max = max(ls_max, max(ls))
                    ls_min = min(ls_min, min(ls))
                    plt.scatter(ts - min(ts), ls, s=10, linewidth=0, c=colors10[count % 10])
                    plot_fitting_curve(ts, ls, colors10[count % 10])
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
                       '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/', '--no-lock'])
        self.assertTrue(r)

    def test_RawLValuesOfSingleMovie(self):
        r = luigi.run(['RawLValuesOfSingleMovie', '--name', '20161013', '--path',
                       '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/MVI_7877.MOV',
                       '--roi', '100,100,20,20', '--no-lock'])
        self.assertTrue(r)

    def test_RawLValuesOfAllMovies(self):
        # shutil.rmtree('data/kinetics/raw/20161013', ignore_errors=True)
        # shutil.rmtree('data/kinetics/raw/20161013 all_l_values.csv', ignore_errors=True)
        r = luigi.run(['RawLValuesOfAllMovies', '--name', '20161013', '--folder',
                       '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/', '--workers', '4', '--no-lock'])
        self.assertTrue(r)

    def test_SplitAllTraces(self):
        shutil.rmtree('data/kinetics/split', ignore_errors=True)
        r = luigi.run(['SplitAllTraces', '--name', '20161013', '--folder',
                       '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/', '--workers', '4', '--no-lock'])
        self.assertTrue(r)

    # def test_PlotSingleKineticsData(self):
    #     shutil.rmtree('data/kinetics/split', ignore_errors=True)
    #     r = luigi.run(['PlotSingleKineticsData', '--name', '20161013', '--folder',
    #                    '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/', '--workers', '1', '--no-lock'])
    #     self.assertTrue(r)

    def test_PlotAllTraces(self):
        shutil.rmtree('dist/kinetics_revision', ignore_errors=True)
        r = luigi.run(['PlotAllTraces', '--name', '20161013', '--folder',
                       '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/', '--workers', '1', '--no-lock'])
        self.assertTrue(r)


#
# Main function
#


class MeasureAndPlotAll(luigi.WrapperTask):
    def requires(self):
        folders = {'20161013': '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/',
                   '20161019': '/Volumes/Mac Ext 2/Suda Electrochromism/20161019/'}
        # folders_test = {'20161013': '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/'}
        tasks = [PlotAllTraces(name=k, folder=v) for k, v in folders.iteritems()]
        for t in tasks:
            yield t


def main():
    # logger = logging.getLogger('luigi-interface')
    # logger.setLevel(logging.WARNING)
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    # folders = {'20161013': '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/',
    #            '20161019': '/Volumes/Mac Ext 2/Suda Electrochromism/20161019/'}
    if os.path.exists('data/kinetics_split/20161013 alldata.p'):
        os.remove('data/kinetics_split/20161013 alldata.p')
    out_folder = 'dist/kinetics_revision'
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    # luigi.run(['MeasureAndPlotAll'])
    luigi.run(['CorrectedLValues', '--name', '20161013', '--folder',
               '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/'])


if __name__ == "__main__":
    unittest.main()
