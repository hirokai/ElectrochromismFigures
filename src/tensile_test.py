import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def main():
    df = pd.read_csv('../data/20161026 tensitle testing.csv', names=list('abcdefghijklmno'), header=0)
    df = df[['a', 'e', 'f', 'i', 'l', 'm']]
    df.columns = ['num', 'pedot', 'rpm', 'w', 'l', 'thick']
    print(df)
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
        print(d)
        l = float(d['l'])
        w = float(d['w'])
        t = float(d['thick'])
        xs = vs[1, :] / l
        ys = smooth(vs[0, :])[5:-5] / (w * t / 1000) * 1e3
        plt.plot(xs, ys)
        plt.xlabel('Strain [-]')
        plt.ylabel('Stress [MPa]')
        plt.xlim([-0.1, 3.1])
        plt.ylim([-0.1, 5])
        plt.title('%02d: %d wt PEDOT, %d rpm' % (i, float(d['pedot']), float(d['rpm'])))
        plt.show()


main()
