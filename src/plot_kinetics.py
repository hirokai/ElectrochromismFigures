import luigi
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from data_tools import colors10, load_csv
from scipy.optimize import curve_fit
from figure_tools import plot_and_save, set_format, set_common_format
from data_tools import split_trace
import seaborn as sns
from pedot_voltage_conditions import CollectCIELabStub
from luigi_tools import cleanup

sns.set_style('white')
sns.set_style("ticks")


def first_order(t, a_i, a_f, k, t0):
    y = a_i + (a_f - a_i) * (1 - np.exp(-k * (t - t0)))
    return y


def plot_20perc(path):
    def func():
        fig, ax = plt.subplots(figsize=(4.5, 3))

        vs = np.array(map(lambda a: [float(a[0]), float(a[2])], load_csv(path)))
        tss, lss = split_trace(np.array(vs[:, 0]), np.array(vs[:, 1]), range(6, 1000, 60))
        count = 0
        # print(tss)
        used_sections_idxs = [5, 7, 9, 11]
        for i, vs in enumerate(zip(tss, lss)):
            ts, ls = vs
            if i in used_sections_idxs and len(ts) > 0:
                ts = ts - min(ts)
                plt.scatter(ts, ls, c=colors10[count % 10], lw=0, s=12)
                fit_start = 2
                popt, _ = curve_fit(first_order, ts[fit_start:60], ls[fit_start:60])
                ts_fit = np.linspace(fit_start, 60, 100)
                ts_fit_plot = np.linspace(1, 60, 100)
                ls_fit_plot = first_order(ts_fit_plot, *popt)
                plt.plot(ts_fit_plot, ls_fit_plot, c=colors10[count % 10], lw=1)
                count += 1
        plt.axis([0, 60, 45, 65])
        set_format(ax, [0, 30, 60], [45, 55, 65], 6, 5)

    return func


def plot_rate_constants_voltage(path):
    plt.figure(figsize=(4.5, 3))
    df = pd.read_csv(path)

    def get(df, v):
        df2 = df[df['PEDOT ratio'] == v][df['skip'] != 1]
        # df2 = pd.concat([df2[df2['voltage'] > 0][df2['mode'] == 'ox'], df2[df2['voltage'] <= 0][df2['mode'] == 'red']])
        df2 = df2[df2['voltage'] > 0][df2['mode'] == 'ox']
        return df2

    ratios = [20, 40, 60, 80]
    # ratios = [20]
    pedots = [get(df, v) for v in ratios]

    def get_exp_with_x0(x0):
        def exp_func(x, A, k):
            return A * np.exp(k * (x - x0))

        return exp_func

    s = """0.11500063
    0.12029616
    0.08706486
    0.12566755"""

    vs = map(float, s.split('\n'))

    count = 0
    for i, pedot in enumerate(pedots):
        # if i == 2:
        #     continue
        plt.scatter(pedot['voltage'], pedot['k'], c=colors10[i], s=25, lw=0)
        exp = get_exp_with_x0(vs[i])
        popt, pcov = curve_fit(exp, pedot['voltage'], pedot['k'], [0.2, 1])
        x = np.linspace(vs[i], 0.8, 100)
        y = exp(x, popt[0], popt[1])
        plt.plot(x, y, c=colors10[i], lw=1)
        count += 1

    plt.axis([0, 1, 0, 0.5])


def plot_rate_constants_pedot(path):
    def func():
        plt.figure(figsize=(4.5, 3))
        df = pd.read_csv(path)

        def get(df, v):
            df2 = df[df['voltage'] == v][df['skip'] != 1]
            df2 = df2[df2['voltage'] > 0][df2['mode'] == 'ox']
            return df2.sort('PEDOT ratio')

        voltages = [0.2, 0.4, 0.6, 0.8]
        pedots = [get(df, v) for v in voltages]

        count = 0
        for i, pedot in enumerate(pedots):
            # if i == 2:
            #     continue
            plt.plot(pedot['PEDOT ratio'], pedot['k'], c=colors10[i], marker='o', mew=0, markersize=5, lw=1)
            count += 1

        plt.axis([0, 100, 0, 1])
    return func


def plot_rate_constants_voltage_red(path):
    def func():
        plt.figure(figsize=(4.5, 3))
        df = pd.read_csv(path)

        def get(df, v):
            df2 = df[df['PEDOT ratio'] == v][df['skip'] != 1]
            df2 = df2[df2['voltage'] <= 0][df2['mode'] == 'red']
            return df2

        ratios = [20, 40, 60, 80]
        pedots = [get(df, v) for v in ratios]

        for i, pedot in enumerate(pedots):
            plt.plot(pedot['voltage'], pedot['k'], c=colors10[i], marker='o', mew=0, markersize=5, lw=1)

        plt.axis([-0.6, 0.2, 0, 0.5])

    return func


class CollectAllKineticsStub(luigi.Task):
    def output(self):
        return luigi.LocalTarget(
            '../../Suda Electrochromism data/20160512-13 Suda EC slices/analysis_0525/curvefit/result.csv')


class PlotOxTrace(luigi.Task):
    name = luigi.Parameter()

    def requires(self):
        return CollectCIELabStub()

    def output(self):
        return [luigi.LocalTarget('../dist/Fig ' + self.name + '.pdf')]

    def run(self):
        set_common_format()
        plot_and_save(plot_20perc(self.input()['15'].path), self.name)


class PlotRateConstants(luigi.Task):
    name1 = luigi.Parameter()
    name2 = luigi.Parameter()

    def requires(self):
        return CollectAllKineticsStub()

    def output(self):
        return [luigi.LocalTarget('../dist/Fig ' + self.name1 + '.pdf'),
                luigi.LocalTarget('../dist/Fig ' + self.name2 + '.pdf')]

    def run(self):
        path = self.input().path
        set_common_format()
        plot_and_save(plot_rate_constants_pedot(path), self.name1)
        plot_and_save(plot_rate_constants_voltage_red(path), self.name2)


class TestPlottingKinetics(luigi.WrapperTask):
    def requires(self):
        yield PlotOxTrace(name='4a')
        yield PlotRateConstants(name1='4c', name2='S5')


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    cleanup(PlotOxTrace(name='4a'))
    cleanup(PlotRateConstants(name1='4c',name2='S5'))
    luigi.run(['TestPlottingKinetics'])
