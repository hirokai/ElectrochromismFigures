import luigi
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import pandas as pd
from data_tools import colors10, load_csv
from scipy.optimize import curve_fit
from figure_tools import plot_and_save, set_format, set_common_format
from data_tools import split_trace
import seaborn as sns
from pedot_voltage_conditions import CollectCIELabStub
from luigi_tools import cleanup


def read_txt(path):
    vss = []
    with open(path) as f:
        while True:
            line = f.readline()
            if line.rstrip() == 'Time/sec\tCurrent/A':
                f.readline()
                break
        for line in f.readlines():
            vss.append(map(float, line.split('\t')))
    return np.array(vss)


def plot_amperometry():
    pedots = [[20, '20160513 Amperometry/80wt% PU const redox.txt'],
              [30, '20160512 Amperometry/70wt% PU const redox.txt'],
              [40, '20160513 Amperometry/60wt% PU const redox.txt'],
              [60, '20160513 Amperometry/40wt% PU const redox.txt'],
              [80, '20160512 Amperometry/20wt% PU const redox.txt'],
              ]
    count = 0
    charges = []
    for pedot, name in pedots:
        vss = read_txt('../../Suda Electrochromism data/%s' % name)
        tss, yss = split_trace(vss[:, 0], vss[:, 1], range(60, 360, 60))
        used_sections_idxs = [2, 4]
        charge = []
        for i, vs in enumerate(zip(tss, yss)):
            if i in used_sections_idxs:
                ts, ys = vs
                ts = ts[3:]
                ys = ys[3:] * 1000  # in mA
                ys_accum = np.add.accumulate(ys) * 0.2
                # print(max(ys_accum))
                charge.append(np.sum(ys) * 0.2)
                plt.plot(ts - min(ts), ys_accum, c=colors10[count % 10])
        charges.append(charge)
        count += 1
    plt.xlim([0, 60])
    plt.show()

    charges = np.array(charges).mean(axis=1)
    print(charges)


def get_amperometry_color():
    current = [[20, '20160513 Amperometry/80wt% PU const redox.txt'],
              [30, '20160512 Amperometry/70wt% PU const redox.txt'],
              [40, '20160513 Amperometry/60wt% PU const redox.txt'],
              [60, '20160513 Amperometry/40wt% PU const redox.txt'],
              [80, '20160512 Amperometry/20wt% PU const redox.txt'],
              ]
    cielab_base = '../../Suda Electrochromism data/20160512-13 Suda EC slices/analysis_0525/cielab/'
    cielab = [[20, '14.csv'],
              [30, '07.csv'],
              [40, '11.csv'],
              [60, '08.csv'],
              [80, '01.csv'],
              ]
    vs = []
    vs1 = []
    for e in zip(current,cielab):
        pedot, name = e[0]
        _, cielab_filename = e[1]
        vss = read_txt('../../Suda Electrochromism data/%s' % name)
        vss_color = load_csv(os.path.join(cielab_base,cielab_filename))
        color_ts = map(lambda r: float(r[0]), vss_color)
        color_ls = map(lambda r: float(r[2]), vss_color)
        tss1, yss1 = split_trace(color_ts,color_ls,range(60, 360, 60))
        ts1 = tss1[4][5:]
        ts1 = ts1 - min(ts1)
        ys1 = yss1[4][5:]


        tss, yss = split_trace(vss[:, 0], vss[:, 1], range(60, 360, 60))
        ts = tss[4][3:]
        ts = ts - min(ts)
        ys = (yss[4][3:]-yss[4][-1]) * 1000
        ys_accum = np.add.accumulate(ys) * 0.2
        vs.append(np.array([ts, ys, ys_accum]).transpose())
        vs1.append(np.array([ts1, ys1]).transpose())
    return vs, vs1


def plot_amperometry_with_color():
    vs,vs1 = get_amperometry_color()
    # print(vs[0][:,1])

    # Plotting accumulated current (=charge)
    charge = []
    for i, v in enumerate(vs):
        plt.plot(v[:, 0], v[:, 1], c=colors10[i])
        plt.plot(v[:, 0], v[:, 2], c=colors10[i])
        charge.append(max(v[:, 2]))

    # Plot color
    color = []
    for i, v in enumerate(vs1):
        plt.plot(v[:, 0], v[:, 1], c=colors10[i],ls='--')
        color.append(max(v[:,1])-min(v[:,1]))

    plt.show()

    vs = np.array([color, charge]).transpose()
    print(vs[:,0])
    plt.scatter(vs[:,0],vs[:,1])
    plt.axis([0,30,0,50])
    plt.show()


class PlotAmperometry(luigi.Task):
    def run(self):
        plot_amperometry_with_color()


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    # cleanup(PlotAmperometry())
    plot_amperometry_with_color()
    set_common_format()
    # luigi.run(['PlotAmperometry'])
