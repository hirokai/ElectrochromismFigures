import os
import csv
import numpy as np
import matplotlib.pyplot as plt

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
        print(vs)
        xs = vs[1, :]
        ys = smooth(vs[0,:])[5:-5]
        print(len(xs), len(ys), len(vs[0, :]))
        plt.plot(xs, ys)
        plt.show()


main()
