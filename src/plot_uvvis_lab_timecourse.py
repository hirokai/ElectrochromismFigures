import luigi
from figure_tools import figure
from data_tools import colors10
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from luigi_tools import cleanup
import re
from figure_tools import set_format, set_common_format
import os


@figure('2b')
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


@figure('3b')
def plot_correlation():
    # Data of redox cycles on 5/23.
    calibration_str = """Abs at 570 nm	Mean L*
    0.249	44.011
    0.472	23.788
    0.278	46.198
    0.403	30.196
    0.253	48.461
    0.439	26.477"""
    calibration = np.array(map(lambda s: map(float, s.split('\t')), calibration_str.split('\n')[1:]))

    plt.figure(figsize=(4.5, 3))
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


@figure('3a')
def plot_timecourse():
    # Data of one redox cycle with 20 sec interval on 5/23.
    time_course_str = """Number	Time [sec]	Abs at 570 nm	Mean L*
    1	0	0.514	17.956
    2	20	0.287	42.205
    3	40	0.272	42.202
    4	60	0.26	45.653
    5	60	0.257	45.668
    6	80	0.415	24.934
    7	100	0.415	18.689
    8	120	0.444	17.943"""
    time_course = np.array(map(lambda s: map(float, s.split('\t')), time_course_str.split('\n')[1:]))

    fig, ax1 = plt.subplots(figsize=(4.5, 3))
    # ax1 = fig.axes()
    ax1.plot(time_course[:, 1], time_course[:, 3], c='b', marker='o', ms=5, mew=0, lw=1)
    plt.xlabel('Time')
    plt.ylabel('Mean L* value')
    plt.ylim([0, 60])

    ax2 = ax1.twinx()
    ax2.plot(time_course[:, 1], time_course[:, 2], c='r', marker='o', ms=5, mew=0, lw=1)
    plt.ylabel('Absorbance at 570 nm')
    plt.ylim([0, 0.6])


class TimeCourseUVStub(luigi.Task):
    def output(self):
        return luigi.LocalTarget('')


class PlotUVVisTimeCIELab(luigi.Task):
    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget('../dist/Fig 2b.pdf'),
                luigi.LocalTarget('../dist/Fig 3a.pdf'),
                luigi.LocalTarget('../dist/Fig 3b.pdf')]

    def run(self):
        set_common_format()
        plot_uvvis()
        plot_timecourse()
        plot_correlation()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    cleanup(PlotUVVisTimeCIELab())
    luigi.run(['PlotUVVisTimeCIELab'])
