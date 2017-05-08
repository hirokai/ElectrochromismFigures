import os
import csv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from common.util import chdir_root, ensure_folder_exists
from common.data_tools import load_csv
from film_thickness.plot_thickness import plot_thickness_rpm_multi

folder = "/Users/hiroyuki/Google Drive/ExpData/Force gauge/20161026 PEDOT-PU/"


# http://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth
def smooth(x, window_len=11, window='hanning'):
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def list_pu_only_files():
    fs = list()
    fs.append("/Users/hiroyuki/Google Drive/ExpData/Force gauge/20170424 PU only/01.csv")
    for i in range(1, 5):
        fs.append("/Users/hiroyuki/Google Drive/ExpData/Force gauge/20170426 PU only/%02d.csv" % i)
    return fs


def draw_pu_only(df):
    chdir_root()
    files = list_pu_only_files()
    plt.subplot(4, 1, 1)
    plt.xlim([-0.1, 3.1])
    plt.ylim([-0.1, 5])
    for i, csv_path in enumerate(files):
        vs = load_csv(csv_path, skip_rows=13, numpy=True)
        xs = vs[:, 1] / 15
        ys = vs[:, 0] / (1e-2 * 3.095e-6) / 1e6  # Thickness is from 2017/4/26 measurement by a surface profiler.

        if i == 0:
            ys = -ys
        if i == 1:
            continue
        xi0 = np.argmax(xs > 0)
        xi1 = np.argmax(xs > 0.05)
        em = (ys[xi1] - ys[xi0]) / (xs[xi1] - xs[xi0])

        winlen_half = 5
        ys = smooth(ys, window_len=winlen_half * 2 + 1)
        ys = ys[winlen_half:-winlen_half]
        ymax = np.max(ys)
        xi = np.argwhere(ys < ymax * 0.1)[-1][0]
        # print(xi,ymax,np.argwhere(ys<ymax*0.1)[-1][0])
        elo = xs[xi]
        pedot = 0
        rpm = 2000
        df = df.append({'i': i, 'pedot': pedot, 'rpm': rpm, 'modulus': em, 'elongation': elo}, ignore_index=True)

        if i in [2, 3]:
            plt.plot(xs, ys, label=os.path.basename(csv_path))
    return df

def main():
    chdir_root()

    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    plt.rcParams['lines.linewidth'] = 1

    df2 = pd.DataFrame()
    df = pd.read_csv('./data/20161026 tensile testing.csv', names=list('abcdefghijklmno'), header=0)
    df = df[['a', 'e', 'f', 'i', 'l', 'm']]
    df.columns = ['num', 'pedot', 'rpm', 'w', 'l', 'thick']
    plt.figure(figsize=(4, 8))
    for i in range(1, 21):
        path = os.path.join(folder, '%02d.csv' % i)
        with open(path) as f:
            vs = []
            reader = csv.reader(f)
            for _ in range(13):
                reader.next()
            for row in reader:
                vs.append(map(float, row))
        vs = np.array(vs).transpose()
        d = df[df['num'] == i]
        l = float(d['l'])
        w = float(d['w'])
        t = float(d['thick'])
        pedot = int(d['pedot'])
        rpm = int(d['rpm'])
        xs = (vs[1, :] - vs[1, 0]) / l
        ys = smooth(vs[0, :])[5:-5] / (w * t / 1000) * 1e3
        ys = ys - ys[0]
        if i == 16:
            ys -= ys[np.argmax(xs >= 0.05)]
            xs -= 0.05
        xi0 = np.argmax(xs > 0)
        xi1 = np.argmax(xs > 0.05)
        em = (ys[xi1] - ys[xi0]) / (xs[xi1] - xs[xi0])
        ymax = np.max(ys)
        xi = np.argwhere(ys < ymax * 0.1)[-1][0]
        # print(xi,ymax,np.argwhere(ys<ymax*0.1)[-1][0])
        elo = xs[xi]
        print(i, pedot, rpm, em, elo)
        df2 = df2.append({'i': i, 'pedot': pedot, 'rpm': rpm, 'modulus': em, 'elongation': elo}, ignore_index=True)
        pos = {0: 1, 20: 2, 30: 3, 40: 4}
        p = pos.get(pedot)
        if rpm != 2000:
            continue
        plt.subplot(4, 1, p)
        plt.plot(xs, ys, label=str(i))
        # plt.legend()
        if p >= 5:
            plt.xlabel('Strain [-]')
        else:
            plt.xlabel('')
        plt.ylabel('Stress [MPa]')
        plt.xlim([-0.1, 3.1])
        plt.ylim([-0.1, 5])
        plt.title('%02d: %d wt PEDOT, %d rpm' % (i, pedot, rpm))

    df2 = draw_pu_only(df2)
    print(df2)
    plt.savefig(os.path.join("formatted", "dist", "Fig S2a.pdf"))
    plt.show()

    plt.figure(figsize=(5, 12))
    plt.subplot(3, 1, 1)
    sns.barplot(x='pedot', y='modulus', data=df2, palette="Set3", capsize=0.1, errwidth=1)
    plt.ylim([0, 40])
    plt.xlabel('')
    plt.ylabel('Elastic modulus [MPa]')

    plt.subplot(3, 1, 2)
    sns.barplot(x='pedot', y='elongation', data=df2, palette="Set3", capsize=0.1, errwidth=1)
    plt.ylim([0, 8])
    plt.xlabel('PEDOT ratio [wt%]')
    plt.ylabel('Elongation at break [-]')

    plt.subplot(3, 1, 3)
    sns.barplot(x='pedot', y='elongation', data=df2, palette="Set3", capsize=0.1, errwidth=1)
    plt.ylim([3, 11])
    plt.xlabel('PEDOT ratio [wt%]')
    plt.ylabel('Elongation at break [-]')

    plt.savefig(os.path.join("formatted", "dist", "Fig S2b.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
