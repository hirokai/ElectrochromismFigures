import luigi
from pedot_voltage_conditions import CurveFitStub
from figure_tools import figure, set_common_format
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_tools import colors10
from scipy.optimize import curve_fit
import os
from luigi_tools import cleanup


@figure('4b')
def plot_final_colors(input_path):
    fig, ax = plt.subplots(figsize=(3, 4.5))
    df = pd.read_csv(input_path)

    # pedot80 = df[df['PEDOT ratio'] == 80]
    # pedot30 = df[df['PEDOT ratio'] == 30]

    def get(df, v):
        df2 = df[df['PEDOT ratio'] == v][df['skip'] != 1]
        df2 = pd.concat([df2[df2['voltage'] > 0][df2['mode'] == 'ox'], df2[df2['voltage'] <= 0][df2['mode'] == 'red']])
        return df2

    ratios = [20, 40, 60, 80]
    pedots = [get(df, v) for v in ratios]

    def sigmoid(x, x0, k, y0, y1):
        y = (y1 - y0) / (1 + np.exp(-k * (x - x0))) + y0
        return y

    popts = []
    perrs = []
    for i, pedot in enumerate(pedots):
        plt.scatter(pedot['voltage'], pedot['L_f'], c=colors10[i])
        popt, pcov = curve_fit(sigmoid, pedot['voltage'], pedot['L_f'], [0, 0.2, min(pedot['L_f']), max(pedot['L_f'])])
        x = np.linspace(-1, 1, 100)
        perr = np.sqrt(np.diag(pcov))
        print popt, perr
        popts.append(popt)
        perrs.append(perr)
        y = sigmoid(x, *popt)
        plt.plot(x, y, c=colors10[i])
        plt.xlim([-0.6, 1])
        # plt.ylim([10, 40])
    ys = [p[0] for p in popts]
    es = [p[0] for p in perrs]
    return ys, es


@figure('S4')
def redox_potentials(ys, es):
    w = 15
    ratios = [20, 40, 60, 80]
    plt.bar(map(lambda a: a - w / 2, ratios), ys, width=w, color='#ccccff')
    plt.errorbar(ratios, ys, es, fmt='none')
    plt.xlim([0, 100])
    plt.ylim([0, 0.2])


class PlotFinalColors(luigi.Task):
    def requires(self):
        return CurveFitStub()

    def output(self):
        return [luigi.LocalTarget('../dist/Fig 4b.pdf'), luigi.LocalTarget('../dist/Fig S4.pdf')]

    def run(self):
        os.chdir(os.path.dirname(__file__))
        ys, es = plot_final_colors(self.input().path)
        set_common_format()
        redox_potentials(ys, es)


class FinalColorsTest(luigi.WrapperTask):
    def requires(self):
        yield PlotFinalColors()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    cleanup(FinalColorsTest())
    luigi.run(['FinalColorsTest'])
