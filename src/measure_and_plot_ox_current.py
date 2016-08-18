import luigi
from figure_tools import plot_and_save, set_common_format
from luigi_tools import cleanup
from data_tools import colors10


# Adopted from: /Users/hiroyuki/Google Drive/ExpData/Potentiostat/20160622 Suda EC/analysis

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
import scipy
from scipy.stats import linregress
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

files = ['500 rpm CV', '750 rpm CV', '1000 rpm CV', '2000 rpm CV',
         '0620/750 rpm cv', '0620/1000 rpm cv', '0620/2000 rpm cv2']


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
            idxs_min.append(i)
        if x == x_max:
            idxs_max.append(i)
    print(idxs_max)
    return idxs_max[-2], idxs_max[-1]


# Find slope for oxidation (y > 0) at voltage x_at
def slope_at(xs, ys, x_at, width=11):
    assert (width % 2 == 1)
    w = int((width - 1) / 2)
    i = np.argwhere(xs == x_at)[1][0]
    print(np.argwhere(xs == x_at), xs[i], ys[i])
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


thickness = [5490, 4377, 3819, 2946, 4377, 3819, 2946]


# http://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
def calc_r2(xdata, ydata, f, popt):
    residuals = ydata - f(xdata, popt[0])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def plot_ox_current_raw_overlay():
    using = ['500 rpm CV','750 rpm CV','0620/2000 rpm cv2']
    # using = [True,True,False,False,False,False,True]
    fig, ax = plt.subplots(figsize=(1.5,0.8))
    xss = []
    yss = []
    count = 0
    for i,n in enumerate(files):
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
        vs_section = vs[:, i_from:i_until]
        xs = vs_section[0]
        ys = vs_section[1]
        xss.append(xs)
        yss.append(ys)
        if n in using:
            plt.plot(xs[1900:2600],1e6*(ys[1900:2600]-ys[1900]),c=colors10[count], label=n, lw=1)
            count += 1
    ax.xaxis.set_major_locator(FixedLocator([-0.2,0.4]))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    plt.ylim([0,40])


def plot_ox_current():
    fig, ax = plt.subplots(figsize=(4.5, 3))
    xss = []
    yss = []
    for n in files:
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
        vs_section = vs[:, i_from:i_until]
        xs = vs_section[0]
        ys = vs_section[1]
        xss.append(xs)
        yss.append(ys)

    currents = []
    thickness_plot = []
    for i, vs in enumerate(zip(xss, yss)):
        xs, ys = vs
        x_from = -0.3
        x_to = 0.6

        slope, intercept = find_aux_line(xs, ys, (-0.2, 0.5), method='connecting')
        xs_aux = np.linspace(x_from, x_to, num=(x_to - x_from) * 1000)
        ys_aux = slope * xs_aux + intercept

        i1 = np.argwhere(xs == x_from)[1][0]
        i2 = np.argwhere(xs == x_to)[1][0]
        if i == 4:  # adhoc fix
            i_max = np.argmax(ys[i1:i2])
            local_max_i = np.array([i_max])
        else:
            local_max_i = find_local_maxima(ys[i1:i2], order=5)[0]
            i_max = local_max_i[0] - 1
        i_max2 = np.argwhere(abs(xs_aux - xs[i1 + i_max]) < 0.0004)[0][0]
        current = ys[i1 + i_max] - ys_aux[i_max2]
        currents.append(current)
        thickness_plot.append(thickness[i])

    used = [True, True, False, True, True, True, True]
    currents = np.array(currents) * 1e6
    print(currents)
    thickness_plot = np.array(thickness_plot) * 0.001
    plt.scatter(thickness_plot, currents, facecolor=map(lambda a: colors10[0] if a else 'r', used),s=25,lw=0)
    used_y = currents[used]
    used_x = thickness_plot[used]

    def func(x,k):
        return k * x

    popt, pcov = curve_fit(func, used_x, used_y)
    slope = popt[0]
    xs = np.linspace(0, 6000, num=2)
    ys = slope * xs
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
        return [luigi.LocalTarget('../dist/Fig '+self.name1+'.pdf'),
                luigi.LocalTarget('../dist/Fig ' + self.name2 + '.pdf')]

    def run(self):
        set_common_format()
        plot_and_save(plot_ox_current,self.name1)
        plot_and_save(plot_ox_current_raw_overlay, self.name2)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    cleanup(PlotOxCurrent(name1='S3',name2='2b-inset'))
    luigi.run(['PlotOxCurrent','--name1','S3','--name2','2b-inset'])
