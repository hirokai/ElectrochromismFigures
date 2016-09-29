import luigi
from figure_tools import plot_and_save
from data_tools import colors10
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from luigi_tools import cleanup
import re
from figure_tools import set_format, set_common_format
import os


def plot_uvvis():
    vs = {}
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for mode in ['OX', 'RE']:
        vs[mode] = []
        for i in [1, 2, 3]:
            path = os.path.join('../../Suda Electrochromism data/20160523/redox cycles/UV-Vis',
                                'PEDOT%s%d.TXT' % (mode, i))
            vs2 = []
            with open(path, 'rU') as f:
                for _ in range(26):
                    f.readline()
                for line in f:
                    ws = re.split('\s+', line)
                    if len(ws) > 3:
                        vs2.append(map(float, [ws[1], ws[3]]))
            vs[mode].append(np.array(vs2))

    vs3 = vs['OX'][0]
    plt.plot(vs3[:, 0], vs3[:, 1], c='#888888', linestyle='--', lw=1)

    vs3 = vs['RE'][0]
    plt.plot(vs3[:, 0], vs3[:, 1], c=colors10[0], lw=1)

    plt.axis([350, 800, 0, 0.8])
    set_format(ax, [400, 500, 600, 700, 800], [0, 0.2, 0.4, 0.6, 0.8], 2, 2)


def plot_correlation():
    # Data of redox cycles on 5/23.
    with open('../data/plot_correlation.txt', 'r') as content_file:
        calibration_str = content_file.read()

    calibration = np.array(map(lambda s: map(float, s.split('\t')), calibration_str.split('\n')[1:]))

    fig = plt.figure(figsize=(4.5, 3))
    ax = fig.add_subplot(111)
    plt.scatter(calibration[:, 0], calibration[:, 1], lw=0, s=25)
    plt.xlabel('Absorbance at 570 nm')
    plt.ylabel('Mean L* value')

    slope, intercept, r_value, p_value, std_err = stats.linregress(calibration[:, 0], calibration[:, 1])
    xs = np.linspace(0, 0.6, 10)
    ys = slope * xs + intercept
    print(slope, intercept)
    plt.plot(xs, ys, lw=1)
    plt.xlim([0, 0.6])
    plt.ylim([0, 60])
    majorLocator = MultipleLocator(0.2)
    minorLocator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    majorLocator = MultipleLocator(20)
    minorLocator = MultipleLocator(10)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)


def plot_timecourse():
    # Data of one redox cycle with 20 sec interval on 5/23.
    with open('../data/plot_timecourse.txt', 'r') as content_file:
        time_course_str = content_file.read()

    time_course = np.array(map(lambda s: map(float, s.split('\t')), time_course_str.split('\n')[1:]))

    fig, ax1 = plt.subplots(figsize=(4.5, 3))
    # ax1 = fig.axes()
    ax1.plot(time_course[:, 1], time_course[:, 3], c='b', marker='o', ms=5, mew=0, lw=1)
    plt.xlabel('Time')
    plt.ylabel('Mean L* value')
    plt.ylim([0, 60])
    majorLocator = MultipleLocator(20)
    minorLocator = MultipleLocator(10)
    ax1.yaxis.set_major_locator(majorLocator)
    ax1.yaxis.set_minor_locator(minorLocator)

    ax2 = ax1.twinx()
    ax2.plot(time_course[:, 1], time_course[:, 2], c='r', marker='o', ms=5, mew=0, lw=1)
    plt.ylabel('Absorbance at 570 nm')
    plt.ylim([0, 0.6])
    majorLocator = MultipleLocator(0.2)
    minorLocator = MultipleLocator(0.1)
    ax2.yaxis.set_major_locator(majorLocator)
    ax2.yaxis.set_minor_locator(minorLocator)


class TimeCourseUVStub(luigi.Task):
    def output(self):
        return luigi.LocalTarget('')


class PlotUVVisTimeCIELab(luigi.Task):
    name1 = luigi.Parameter()
    name2 = luigi.Parameter()
    name3 = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget('../dist/Fig ' + self.name1 + '.pdf'),
                luigi.LocalTarget('../dist/Fig ' + self.name2 + '.pdf'),
                luigi.LocalTarget('../dist/Fig ' + self.name3 + '.pdf')]

    def run(self):
        set_common_format()
        plot_and_save(plot_uvvis, self.name1)
        plot_and_save(plot_timecourse, self.name2)
        plot_and_save(plot_correlation, self.name3)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    cleanup(PlotUVVisTimeCIELab(name1='2a', name2='3a', name3='3b'))
    luigi.run(['PlotUVVisTimeCIELab', '--name1', '2b', '--name2', '3a', '--name3', '3b'])
