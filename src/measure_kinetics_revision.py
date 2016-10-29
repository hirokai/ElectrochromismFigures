# -*- coding: utf-8 -*-

import os
from subprocess import call
import csv
import sys
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
from data_tools import colors10, split_trace
from scipy.optimize import curve_fit


folder = '/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/'
roi_path = '/Users/hiroyuki/Documents/Nishizawa Lab/2016 準備中の論文/20160316 Paper Suda electrochromism/20161017 測定項目/rois.csv'
names_path = ''


def measure_cielab(path, rois):
    rgb_whole = io.imread(path)
    ms = []
    for roi in rois:
        rgb = rgb_whole[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        lab = color.rgb2lab(rgb)
        m = np.mean(lab[:, :, 0])
        ms.append(m)
    return ms


def measure_movie_slices(subfolder, roi_samples, max_timepoints=1000, rois_calibration=None):
    print('Measuring slices: ' + subfolder)
    folder_path = os.path.join(folder, subfolder)
    files = sorted(os.listdir(folder_path))
    lss = np.zeros((3, max_timepoints))
    for i, name in enumerate(files):
        path = os.path.join(folder, subfolder, name)
        sys.stdout.write('.')
        sys.stdout.flush()
        ls = measure_cielab(path, roi_samples)
        lss[:, i] = ls
    sys.stdout.write('\n')
    return np.array(lss)


def mk_slices(path):
    out_folder = path + '_out'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_path = os.path.join(out_folder, "out%04d.png")
    print(out_path)
    cmd = ["/Users/hiroyuki/bin/ffmpeg", "-i", path, "-vf", "fps=1", out_path]
    print(cmd)
    call(cmd)


def all_make_slices():
    files = filter(lambda n: n.find('.MOV') >= 0, os.listdir(folder))
    print(files)
    for n in files:
        path = os.path.join(folder, n)
        if os.path.isfile(path):
            mk_slices(path)


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


def all_measure_cielab():
    folders = sorted(filter(lambda n: os.path.isdir(os.path.join(folder, n)), os.listdir(folder)))
    rois = read_rois(roi_path)
    lsss = []
    max_timepoints = 1000
    for f in folders:
        num = f[4:8]
        lss = measure_movie_slices(f, rois.get(num), max_timepoints=max_timepoints)
        lsss.append(lss)
    result = np.array(lsss).transpose()
    print(result.shape)
    result = np.reshape(result, (max_timepoints, len(folders) * 3))
    print(result.shape)
    np.savetxt(os.path.join(folder, 'all_l_values.csv'), result, delimiter=",")


def organize_data():
    with open(os.path.join(folder, 'all_l_values.csv')) as f:
        reader = csv.reader(f)
        rows = np.array([map(float, r) for r in reader])
    dat = {}
    count = 0

    with open(os.path.join(folder, 'sample_conditions.csv')) as f:
        reader = csv.reader(f)
        reader.next()

        def func(row):
            return row[1] + ' wt% PEDOT, ' + row[0] + 'rpm'

        sample_names = [func(r) for r in reader]

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
        row_truncated = rows[:, i][0:np.argmin(rows[:, i])]
        dat[series][pos] = row_truncated
        count += 1
    return dat, sample_names


def all_plot(dat, sample_names, save=False):
    size = (14, 3)
    for i in range(30):
        plt.figure(figsize=size)
        for j in range(3):
            ys = dat[i][j]
            xs = np.array(range(len(ys))) + (j * 800)
            print(i, j, len(xs))
            plt.plot(xs, ys, c=colors10[0])
        plt.ylim([0, 60])
        plt.title('Sample #%d: %s' % (i + 1, sample_names[i]))
        if save:
            plt.savefig('%s.png' % sample_names[i])
        plt.show()


def first_order(t, a_i, a_f, k, t0):
    y = a_i + (a_f - a_i) * (1 - np.exp(-k * (t - t0)))
    return y


def print_fit(ts, ys, c):
    if len(ts) < 46:
        return
    fit_start = 5
    try:
        popt, _ = curve_fit(first_order, ts[fit_start:45], ys[fit_start:45],[ys[fit_start], ys[45], 0.1, 2])
        ts_fit = np.linspace(fit_start, 45, 100)
        ts_fit_plot = np.linspace(fit_start, 45, 100)
        ys_fit_plot = first_order(ts_fit_plot, *popt)
        plt.plot(ts_fit_plot, ys_fit_plot, c=c, lw=1)
    except:
        pass


def plot_split_traces(dat, sample_names):
    plt.figure(figsize=(20, 15))
    for i in range(30):
        v = dat[i][0]
        t = range(len(v))
        plt.subplot(6, 3, (i % 6) * 3 + 1)
        tss, vss = split_trace(t, v, range(2, 1000, 60))
        count = 0
        for ts, vs in zip(tss, vss):
            if len(ts) > 0:
                print_fit(ts, vs, colors10[count % 10])
                plt.scatter(ts - min(ts), vs, c=colors10[count % 10], linewidth=0)
            count += 1
        plt.title('Const: ' + sample_names[i])
        plt.ylim([0, max(v)])

        v = dat[i][1]
        t = range(len(v))
        plt.subplot(6, 3, (i % 6) * 3 + 2)
        tss, vss = split_trace(t, v, range(2, 1000, 60))
        count = 0
        for ts, vs in zip(tss[1::2], vss[1::2]):
            if len(ts) > 0:
                print_fit(ts, vs, colors10[count % 10])
                plt.scatter(ts - min(ts), vs, c=colors10[count % 10], linewidth=0)
            count += 1
        plt.title('Ox: ' + sample_names[i])
        plt.ylim([0, max(v)])

        v = dat[i][2]
        t = range(len(v))
        plt.subplot(6, 3, (i % 6) * 3 + 3)
        tss, vss = split_trace(t, v, range(2, 1000, 60))
        count = 0
        for ts, vs in zip(tss[1::2], vss[1::2]):
            if len(ts) > 0:
                print_fit(ts, vs, colors10[count % 10])
                plt.scatter(ts - min(ts), vs, c=colors10[count % 10], linewidth=0)
            count += 1
        plt.title('Red: ' + sample_names[i])
        plt.ylim([0, max(v)])
        if i % 6 == 5:
            plt.show()
            plt.figure(figsize=(20, 15))


def main():
    # all_make_slices() -> Done
    # all_measure_cielab() -> Done
    dat, names = organize_data()
    plot_split_traces(dat, names)
    # all_plot(dat, names)


if __name__ == "__main__":
    main()