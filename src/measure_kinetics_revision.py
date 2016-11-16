import os
from subprocess import call
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from data_tools import colors10, split_trace
from scipy.optimize import curve_fit
from util import ensure_exists, ensure_folder_exists, basename_noext, bcolors
from image_tools import get_cie_l_rois
from test_calibration_fit import correct_cielab
import luigi
from luigi_tools import cleanup
import shutil
import cPickle as pickle
import logging
from make_slices import mk_slices


#
# Handling movies, conditions, ROIs
#

def mk_condition_name(c):
    return '%d wt%% PEDOT, %d rpm' % (c['pedot'], c['rpm'])


def read_condition_data(path):
    with open(path) as f:
        reader = csv.reader(f)
        reader.next()
        sample_conditions = [{'pedot': int(row[2]), 'rpm': int(row[1])} for row in reader]
    return sample_conditions


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

class MakeAllSlices(luigi.WrapperTask):
    folder = luigi.Parameter()

    def requires(self):
        folder_slices = os.path.join(self.folder, 'slices')
        folders = sorted(filter(lambda n: os.path.isdir(os.path.join(folder_slices, n)), os.listdir(folder_slices)))
        roi_path = 'parameters/%s/rois.csv' % self.name
        rois = read_rois(roi_path)
        print(sorted(rois.keys()), folders)
        print(rois)
        for num, f in enumerate(folders):
            yield MakeSingleMovieSlices(path=f)


class MakeSingleMovieSlices(luigi.Task):
    path = luigi.Parameter()

    def run(self):
        print(self.path)
        assert (os.path.isfile(self.path))
        print(bcolors.OKGREEN + 'MakeSingleMovieSlices: \nInput %s' % self.path + bcolors.ENDC)
        print(bcolors.OKGREEN + 'Output: %s' % self.output().path + bcolors.ENDC)
        mk_slices(self.path)

    def output(self):
        return luigi.LocalTarget(os.path.join(os.path.dirname(self.path), 'slices', basename_noext(self.path)))


#
# Measurement of slices
#

def measure_movie_slices(folder_path, roi_samples, max_timepoints=1000, rois_calibration=None):
    print('Measuring slices: ' + folder_path)
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
    folder = luigi.Parameter()
    roi = luigi.Parameter()

    def requires(self):
        return MakeSingleMovieSlices(path=self.folder)

    def run(self):
        print('%s' % self.folder)
        lss = measure_movie_slices(self.folder, self.roi)
        np.savetxt(self.output().path, lss)
        # with open(self.output().path) as f:
        #     writer = csv.writer(f)
        #     writer.writerow([for ls in lss])

    def output(self):
        num = os.path.basename(self.folder)[4:8]
        return luigi.LocalTarget(os.path.join('data', 'kinetics', 'raw', self.name, "%s.csv" % num))


class RawLValues(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        files = filter(lambda n: os.path.isdir(os.path.join(self.folder, 'slices', n.split('.')[0])),
                       sorted(os.listdir(str(self.folder))))
        print('RawLValues: ' + ','.join(files))
        roi_path = 'parameters/%s/rois.csv' % self.name
        rois = read_rois(roi_path)
        return [RawLValuesOfSingleMovie(name=self.name, folder=os.path.join(self.folder, f), roi=rois.get(f[4:8]))
                for f in files]

    def output(self):
        return luigi.LocalTarget(os.path.join('data', 'kinetics', '%s all_l_values.csv' % self.name))

    def run(self):
        max_timepoints = 1000
        raw_values = self.input()
        vss = np.zeros((len(raw_values), max_timepoints))
        # vs = np.zeros((1,max_timepoints))
        for i, raw in enumerate(raw_values):
            vs1 = [float(r[1]) for r in raw.path]
            # vs[0:(len(vs1))] = vs1
            vss[i, 0:(len(vs1))] = vs1
        print('RawLValues: ', self.name, self.folder, raw_values)
        print(vss)
        np.savetxt(self.output().path, vss)


#
# Correction by color chart
#

def correct_cielab_stub(in_csv, scale_csv, out_csv):
    import shutil
    shutil.copyfile(in_csv, out_csv)


class CorrectedLValues(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return RawLValues(name=self.name, folder=self.folder)

    def output(self):
        return luigi.LocalTarget(
            os.path.join('data', 'kinetics', '%s all_l_values_corrected_fixme.csv' % self.name))

    def run(self):
        correct_cielab_stub(self.input().path,
                            os.path.join('data', 'kinetics', '%s calibration scale.txt' % self.name),
                            self.output().path)


#
# Splitting traces
#

class SplitKineticsData(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return CorrectedLValues(name=self.name, folder=self.folder)

    def run(self):
        in_csv = os.path.join('data', 'kinetics', '%s all_l_values.csv' % self.name)
        conditions_csv = os.path.join('parameters', self.name, 'sample_conditions.csv')
        dat, conditions = organize_data(in_csv, conditions_csv)
        split_dat = split_all_traces(dat, conditions)
        print(split_dat)
        save_split_data(split_dat, self.name, self.output().path)

    def output(self):
        return luigi.LocalTarget(os.path.join('data', 'kinetics_split', '%s alldata.p' % self.name))


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

    sample_conditions = read_condition_data(conditions_path)

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


def split_all_traces(dat, conditions):
    res = SplitTraces()
    for i in range(30):
        cond = conditions[i]
        pedot = cond['pedot']
        rpm = cond['rpm']

        # Const
        v = dat[i][0]
        t = range(len(v))
        tss, vss = split_trace(t, v, range(2, 1000, 60))
        for i, ys in enumerate(zip(tss[1::2], vss[1::2])):
            voltages = [0.8, -0.5, 0.8, -0.5, 0.8, -0.5]  # FIXME: Check values.
            if i >= len(voltages):
                break
            ts = ys[0]
            vs = ys[1]
            voltage = voltages[i]
            res.set_data(pedot, rpm, 'const', voltage, ts, vs)

        # Ox
        v = dat[i][1]
        t = range(len(v))
        tss, vss = split_trace(t, v, range(2, 1000, 60))
        for i, ys in enumerate(zip(tss[1::2], vss[1::2])):
            voltages = [0, 0.2, 0.4, 0.6, 0.8]  # FIXME: Check values.
            if i >= len(voltages):
                break
            ts = ys[0]
            vs = ys[1]
            voltage = voltages[i]
            res.set_data(pedot, rpm, 'ox', voltage, ts, vs)

        # Red
        v = dat[i][2]
        t = range(len(v))
        tss, vss = split_trace(t, v, range(2, 1000, 60))
        for i, ys in enumerate(zip(tss[1::2], vss[1::2])):
            voltages = [0.1, -0.1, -0.3, -0.5, -0.7]  # FIXME: Check values.
            if i >= len(voltages):
                break
            ts = ys[0]
            vs = ys[1]
            voltage = voltages[i]
            res.set_data(pedot, rpm, 'red', voltage, ts, vs)
    return res


def save_split_data(dat, dataset_name, p_path):
    assert isinstance(dat, SplitTraces)
    print(dat)
    for k, d in dat.dat.iteritems():
        pedot, rpm, mode, voltage = SplitTraces.get_cond_from_key(k)
        out_path = os.path.join('data', 'kinetics_split', dataset_name,
                                '%d perc PEDOT - %d rpm' % (pedot, rpm),
                                '%s %.1f.csv' % (mode, voltage))
        ts, vs = d
        ensure_folder_exists(out_path)
        rows = map(list, zip(*[[str(t) for t in ts], [str(v) % v for v in vs]]))
        with open(out_path, 'wb') as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)
    with open(p_path, 'wb') as f:
        pickle.dump(dat, f)


def read_all_split_traces(path):
    with open(path) as f:
        obj = pickle.load(f)
    return obj


#
# Functions for first-oder fitting.
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
        popt, _ = curve_fit(first_order, ts[fit_start:fit_to], ys[fit_start:fit_to],
                            [ys[fit_start], ys[fit_to], 0.1, 2], method='lm')
        # ts_fit = np.linspace(fit_start, fit_to, 100)
        ts_fit_plot = np.linspace(fit_start, 45, 100)
        ys_fit_plot = first_order(ts_fit_plot, *popt)
        plt.plot(ts_fit_plot, ys_fit_plot, c=c, lw=1)
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
    plt.figure(figsize=(20, 15))
    for i in range(30):
        name = mk_condition_name(sample_conditions[i])
        # Constant voltages.
        tss, vss = dat.get_data('')[i]['const']
        plt.subplot(6, 3, (i % 6) * 3 + 1)
        count = 0
        for ts, vs in zip(tss, vss):
            if len(ts) > 0:
                popt = plot_fitting_curve(ts, vs, colors10[count % 10])
                # print(popt)
                plt.scatter(ts - min(ts), vs, c=colors10[count % 10], linewidth=1)
            count += 1
        plt.title('Const: ' + name)
        # plt.ylim([min(v), max(v)])

        # Varied oxidation voltages.
        tss, vss = dat[i]['ox']
        plt.subplot(6, 3, (i % 6) * 3 + 2)
        count = 0
        for ts, vs in zip(tss, vss):
            if len(ts) > 0:
                popt = plot_fitting_curve(ts, vs, colors10[count % 10])
                plt.scatter(ts - min(ts), vs, c=colors10[count % 10], linewidth=1)
                # print(popt)
            count += 1
        plt.title('Ox: ' + name)
        # plt.ylim([min(v), max(v)])

        # Varied reduction voltages.
        tss, vss = dat[i]['red']
        plt.subplot(6, 3, (i % 6) * 3 + 3)
        count = 0
        for ts, vs in zip(tss, vss):
            if len(ts) > 0:
                popt = plot_fitting_curve(ts, vs, colors10[count % 10])
                # print(popt)
                plt.scatter(ts - min(ts), vs, c=colors10[count % 10], linewidth=1)
            count += 1
        plt.title('Red: ' + name)
        # plt.ylim([min(v), max(v)])
        if i % 6 == 5:
            if save_folder:
                plt.savefig(os.path.join(save_folder, '%02d.pdf' % i))
            else:
                plt.show()
            plt.figure(figsize=(20, 15))


def plot_single_split_trace(dataset_name, pedot, rpm, mode, idx):
    path = os.path.join('data', 'kinetics_split', dataset_name, '%d perc PEDOT - %d rpm' % (pedot, rpm),
                        '%s %d.csv' % (mode, idx))
    print('Plotting: %s' % path)
    with open(path) as f:
        reader = csv.reader(f)
        vs = np.array([map(float, r) for r in reader])
    plt.plot(vs[:, 0], vs[:, 1])
    plt.show()


def plot_single_split_trace2(dataset_name, pedot, rpm, mode, voltage):
    path = os.path.join('data', 'kinetics_split', dataset_name, '%d perc PEDOT - %d rpm' % (pedot, rpm),
                        '%s %.1f.csv' % (mode, voltage))
    print('Plotting: %s' % path)
    with open(path) as f:
        reader = csv.reader(f)
        vs = np.array([map(float, r) for r in reader])
    plt.plot(vs[:, 0], vs[:, 1])
    plt.show()


class PlotKineticsData(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return SplitKineticsData(name=self.name, folder=self.folder)

    def run(self):
        conditions_csv = os.path.join('parameters', self.name, 'sample_conditions.csv')
        conditions = read_condition_data(conditions_csv)
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
    index = luigi.IntParameter()

    def requires(self):
        return SplitKineticsData(name=self.name, folder=self.folder)

    def run(self):
        split_dat = read_all_split_traces('data/kinetics_split/%s alldata.p' % self.name)
        for pedot in [20, 30, 40]:
            plot_single_split_trace2(self.name, pedot, 1000, 'const', 0.8)
        for rpm in [500, 1000, 2000]:
            plot_single_split_trace2(self.name, 30, rpm, 'ox', 0.8)


class PlotAllKineticsData(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        return SplitKineticsData(name=self.name, folder=self.folder)

    def run(self):
        conditions_csv = os.path.join('parameters', self.name, 'sample_conditions.csv')
        conditions = read_condition_data(conditions_csv)
        split_dat = read_all_split_traces('data/kinetics_split/%s alldata.p' % self.name)
        save_folder = os.path.join('dist', 'kinetics_revision', self.name)
        ensure_exists(save_folder)
        plot_split_traces(split_dat, conditions, save_folder=save_folder)
        all_plot(split_dat, conditions, save_folder=save_folder)

    def output(self):
        return luigi.LocalTarget('dist/kinetics_revision')


#
# Main function
#

class MeasureAndPlotAll(luigi.WrapperTask):
    def requires(self):
        folders = {'20161013': '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/',
                   '20161019': '/Volumes/Mac Ext 2/Suda Electrochromism/20161019/'}
        # folders_test = {'20161013': '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/'}
        tasks = [PlotAllKineticsData(name=k, folder=v) for k, v in folders.iteritems()]
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
    main()
