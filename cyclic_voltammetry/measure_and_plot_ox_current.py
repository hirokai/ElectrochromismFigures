import csv
import os
import re

import luigi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.stats import linregress

from src.data_tools import colors10
from src.luigi_tools import cleanup
from src.figure_tools import plot_and_save, set_common_format


def split_cv_cycle(xs, x_min=None, x_max=None):
    if x_min is None:
        x_min = min(xs)
    if x_max is None:
        x_max = max(xs)
    assert (x_min < x_max)
    idxs_min = []
    idxs_max = []
    for i, x in enumerate(xs):
        if x == x_min:
            if len(idxs_min) == 0 or i - idxs_min[-1] > 1:  # 2 successive points with same value are ignored.
                idxs_min.append(i)
        if x == x_max:
            if len(idxs_max) == 0 or i - idxs_max[-1] > 1:  # 2 successive points with same value are ignored.
                idxs_max.append(i)
    return idxs_max[-2], idxs_max[-1]


# Find slope for oxidation (y > 0) at voltage x_at
def slope_at(xs, ys, x_at, width=11):
    assert (width % 2 == 1)
    w = int((width - 1) / 2)
    i = np.argwhere(xs == x_at)[1][0]
    # print(np.argwhere(xs == x_at), xs[i], ys[i])
    slope, intercept, r_value, p_value, std_err = linregress(xs[i - w:i + w + 1], ys[i - w:i + w + 1])
    return slope, i


def find_aux_line(xs, ys, arg=None, method='connecting'):
    if method == 'slope_at':
        x_at = arg or 0.1
        slope, j = slope_at(xs, ys, x_at)
        return slope, (ys[j] - slope * x_at)
    elif method == 'connecting':
        arg = arg or (-0.2, 0.5)
        x_from, x_to = arg
        i1 = np.argwhere(xs == x_from)[1][0]
        i2 = np.argwhere(xs == x_to)[1][0]
        slope = (ys[i2] - ys[i1]) / (x_to - x_from)
        intercept = ys[i1] - slope * x_from
        return slope, intercept


def find_local_maxima(vs, order=5):
    return argrelextrema(vs, np.greater, order=order)


# http://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
def calc_r2(xdata, ydata, f, popt):
    residuals = ydata - f(xdata, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def plot_ox_current_raw_overlay():
    using = ['500 rpm CV', '750 rpm CV', '0620/2000 rpm cv2']
    # using = [True,True,False,False,False,False,True]
    fig, ax = plt.subplots(figsize=(1.5, 0.8))
    xss = []
    yss = []
    count = 0

    files = [
        "71 30wt 1krpm.txt", "72 30wt 1krpm.txt", "74 30wt 2krpm.txt",
        "86 30wt 3krpm.txt", "87 30wt 3krpm.txt", "88 30wt 4krpm.txt",
        "89 30wt 4krpm.txt", "90 30wt 5krpm.txt", "91 30wt 5krpm.txt"]

    for i, n in enumerate(files):
        path = os.path.join('..', 'data', 'cv', n + '.txt')
        with open(path) as f:
            while True:
                l = f.readline()
                if l.find('Potential/V') == 0:
                    break
            f.readline()
            reader = csv.reader(f, delimiter='\t')
            vs = np.transpose(map(lambda row: map(float, row)[0:2], [r for r in reader]))
        i_from, i_until = split_cv_cycle(vs[0])
        # print(i_from, i_until)
        vs_section = vs[:, i_from:i_until]
        xs = vs_section[0]
        ys = vs_section[1]
        xss.append(xs)
        yss.append(ys)
        if n in using:
            plt.plot(xs[1900:2600], 1e6 * (ys[1900:2600] - ys[1900]), c=colors10[count], label=n, lw=1)
            count += 1
    ax.xaxis.set_major_locator(FixedLocator([-0.2, 0.4]))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    plt.ylim([0, 40])


def read_raw_data(files, thickness_table):
    data_series = []
    instrument = None
    for n in files:
        path = os.path.join('data', 'cv', n)
        with open(path) as f:
            while True:
                l = f.readline()
                if l.find('Potential/V') == 0:
                    break
                m = re.search(r"Instrument Model:\s+(\S+)", l)
                if m:
                    instrument = m.group(1)
            for _ in range(2 if instrument == 'ALS2323' else 1):
                f.readline()
            rows = []
            for l in f.readlines():
                if l.strip() != '':
                    rows.append(map(float, l.split(', ' if instrument == 'ALS2323' else '\t')))
            vs = np.transpose(rows)
        i_from, i_until = split_cv_cycle(vs[0])
        vs_section = vs[:, i_from:i_until]
        xs = vs_section[0]
        ys = vs_section[1]
        print(n)
        rpm = re.search(r"[/ ](\w+)\s?rpm", n).group(1)
        idx = thickness_table.keys().index(rpm)
        data_series.append([n, rpm, thickness_table[rpm], xs, ys, colors10[idx]])
    return data_series


def measure_and_plot_ox_current():
    thickness_table_0620 = {'500': 5490, '750': 4377, '1000': 3819, '2000': 2946}

    thickness_table = {'500': 5317, '1k': 3961, '2k': 2936, '3k': 2717, '4k': 2662, '5k': 2434}

    fig, ax = plt.subplots(figsize=(4.5, 3))

    files_0620 = ['0622/500 rpm CV.txt', '0622/750 rpm CV.txt', '0620/750 rpm cv.txt',
                  '0622/1000 rpm CV.txt', '0620/1000 rpm cv.txt',
                  '0622/2000 rpm CV.txt', '0620/2000 rpm cv2.txt']
    files = [
        "1021/71 30wt 1krpm.txt", "1021/72 30wt 1krpm.txt", "1021/74 30wt 2krpm.txt", "0620/30wt 2krpm.txt",
        "1021/86 30wt 3krpm.txt", "1021/87 30wt 3krpm.txt", "1021/88 30wt 4krpm.txt", "1021/89 30wt 4krpm.txt"]

    data_series = read_raw_data(files_0620, thickness_table_0620) + read_raw_data(files, thickness_table)

    # Measure current
    results = []
    x_from = -0.3
    x_to = 0.5
    for i, vs in enumerate(data_series):
        n, rpm, thickness, xs, ys, _ = vs

        slope, intercept = find_aux_line(xs, ys, (-0.2, 0.5), method='connecting')
        xs_aux = np.linspace(x_from, x_to, num=(x_to - x_from) * 1000)
        ys_aux = slope * xs_aux + intercept

        i1 = np.argwhere(xs == x_from)[1][0]
        # i2 = np.argwhere(xs == x_to)[1][0]
        i2 = np.argwhere(xs == x_to)[1][0]

        # Measurement of current using the auxiliary line calculated above.
        x_max_point = xs[np.argmax(ys[i1:i2]) + i1]
        y_max_point = ys[np.argmax(ys[i1:i2]) + i1]
        current = y_max_point - ys_aux[np.argmax(ys[i1:i2]) - 2]
        results.append([slope, intercept, current, x_max_point, y_max_point, ys_aux[np.argmax(ys[i1:i2]) - 2]])

    # Plotting each measurement result
    color_table = {'500': colors10[0], '750': colors10[1], '1000': colors10[2], '1k': colors10[2],
                   '2000': colors10[3], '2k': colors10[3], '3000': colors10[4], '3k': colors10[4],
                   '4000': colors10[5], '4k': colors10[5]}
    for i, vs in enumerate(zip(data_series, results)):
        vs1, vs2 = vs
        n, rpm, thickness, xs, ys, _ = vs1
        slope, intercept, current, x_max_point, y_max_point, yaux_max_point = vs2
        color = color_table[rpm]
        plt.subplot(5, 4, i + 1)
        plt.plot(xs, ys, c=color)
        xs_aux = np.linspace(x_from, x_to, num=(x_to - x_from) * 1000)
        ys_aux = slope * xs_aux + intercept
        line, = plt.plot(xs_aux, ys_aux, label='%s rpm' % rpm, c=color)
        plt.plot([x_max_point, x_max_point], [yaux_max_point, y_max_point])
        plt.ylim([-1e-4, 1e-4])
        plt.title('%s rpm, %.1e' % (rpm, current))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.show()

    # Plotting current versus thickness
    currents = np.array(map(lambda a: a[2], results)) * 1e6  # in microA
    thickness_for_plot = np.array(map(lambda a: a[2], data_series)) * 1e-3  # in um
    used = np.array([True] * currents.shape[0])
    # used[3] = False
    plt.scatter(thickness_for_plot, currents, facecolor=map(lambda a: colors10[0] if a else 'r', used), s=25, lw=0)
    used_y = currents[used]
    used_x = thickness_for_plot[used]

    def func(x, k):
        x0 = 0
        return k * x + x0

    popt, pcov = curve_fit(func, used_x, used_y)
    slope = popt[0]
    # x0 = popt[1]
    xs = np.linspace(0, 6000, num=2)
    ys = func(xs,*popt)
    r2 = calc_r2(used_x, used_y, func, popt)
    print(slope,r2)
    plt.plot(xs, ys, c=colors10[0], lw=1)
    plt.xlabel('Film thickness [um]')
    plt.ylabel('Oxidative peak current [uA]')
    plt.xlim([0, 6])
    plt.ylim([0, 30])
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))


class PlotOxCurrent(luigi.Task):
    name1 = luigi.Parameter()
    name2 = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget('./dist/Fig ' + self.name1 + '.pdf'),
                luigi.LocalTarget('./dist/Fig ' + self.name2 + '.pdf')]

    def run(self):
        set_common_format()
        plot_and_save(measure_and_plot_ox_current, self.name1)
        plot_and_save(plot_ox_current_raw_overlay, self.name2)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__),os.pardir))
    # cleanup(PlotOxCurrent(name1='S3',name2='2b-inset'))
    # luigi.run(['PlotOxCurrent','--name1','S3','--name2','2b-inset'])
    plot_and_save(measure_and_plot_ox_current, 'cv_thickness_revision',show=True)
    # cleanup(PlotOxCurrent(name1='cv_thickness_revision', name2='cv_thickness_revision_inset'))
    # luigi.run(['PlotOxCurrent', '--name1', 'cv_thickness_revision', '--name2', 'cv_thickness_revision_inset'])
