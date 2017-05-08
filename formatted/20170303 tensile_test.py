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
    df2 = pd.DataFrame()
    df = pd.read_csv('./20161026 tensile testing.csv', names=list('abcdefghijklmno'), header=0)
    df = df[['a', 'e', 'f', 'i', 'l', 'm']]
    df.columns = ['num', 'pedot', 'rpm', 'w', 'l', 'thick']
    plt.figure(figsize=(10, 6))
    for i in range(3, 21):
        if i == 10:
            continue
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
        # print(i, pedot, rpm, em)
        ymax = np.max(ys)
        xi = np.argwhere(ys<ymax*0.1)[-1][0]
        # print(xi,ymax,np.argwhere(ys<ymax*0.1)[-1][0])
        elo = xs[xi]
        df2 = df2.append({'i': i, 'pedot': pedot, 'rpm': rpm, 'modulus': em, 'elongation': elo}, ignore_index=True)
        pos = {20: {500: 1, 2000: 2}, 30: {500: 3, 2000: 4}, 40: {500: 5, 2000: 6}}
        p = (pos.get(pedot) or {}).get(rpm) or 1
        plt.subplot(3, 2, p)
        plt.plot(xs, ys, label=str(i))
        plt.legend()
        if p >= 5:
            plt.xlabel('Strain [-]')
        else:
            plt.xlabel('')
        plt.ylabel('Stress [MPa]')
        plt.xlim([-0.1, 3.1])
        plt.ylim([-0.1, 5])
        plt.title('%02d: %d wt PEDOT, %d rpm' % (i, pedot, rpm))
    # plt.savefig('20170303 stress strain.pdf')
    plt.show()
    gr = df2.groupby('pedot')
    print(gr.mean()['modulus'],gr.std()['modulus'])
    print(gr.mean()['elongation'],gr.std()['elongation'])
    import seaborn as sns
    # sns.swarmplot(x='pedot',y='modulus',hue='rpm',data=df2,size=10,edgecolor=None)
    print(df2)
    # print(df2['modulus'])
    plt.figure(figsize=(5, 6))
    plt.subplot(2,1,1)
    sns.barplot(x='pedot',y='modulus',data=df2,palette="Set3",capsize=0.1, errwidth=1)
    plt.ylim([0,40])
    plt.xlabel('')
    plt.ylabel('Elastic modulus [MPa]')
    plt.subplot(2,1,2)
    sns.barplot(x='pedot',y='elongation',data=df2,palette="Set3",capsize=0.1, errwidth=1)
    plt.ylim([0,4])
    plt.xlabel('PEDOT ratio [wt%]')
    plt.ylabel('Elongation at break [-]')
    plt.savefig('20170303 elastic moduli and elongation at break.pdf')
    plt.show()


main()
