import fnmatch
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from common.util import ensure_folder_exists, chdir_root

from common.data_tools import load_csv, save_csv


def get_out_path(path):
    return path.replace('/split/', '/fitted_manual/')


# Specify conditions chosen for analysis
def recursive_walk(folder):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*.csv'):
            matches.append(os.path.join(root, filename))
    return [m for m in matches if
            # m.find('red') != -1
            # and
            m.find('2000 rpm') != -1
            # # m.find('0.8') != -1 and
            # # not m.find('const -0.5') != -1 and
            and not os.path.exists(get_out_path(m))
            ]


skinv = None
st0 = None
sli = None
slf = None
l_scatter = None
ax = None
l = None
paths = []
count = 0


def main(name):
    global skinv, st0, sli, slf, l_scatter, fig, ax, l, paths, count

    plt.interactive(False)
    count = 0
    paths = recursive_walk('./data/kinetics/split/%s' % name)
    if not paths:
        print('No files. Quitting.')
        return

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    vs = load_csv(paths[0], numpy=True)
    ts_dat = vs[:, 0]
    ys_dat = vs[:, 1]
    l_scatter, = plt.plot(ts_dat - min(ts_dat), ys_dat, marker='o', markersize=3, lw=0)

    t00 = 2
    kinv0 = 5
    li0 = 10
    lf0 = 50
    t = np.arange(t00, 60, 0.1)

    s = li0 + (lf0 - li0) * (1 - np.exp(-(t - t00) / kinv0))

    l, = plt.plot(t, s, lw=2, color='red')

    axcolor = 'lightgoldenrodyellow'
    axlf = plt.axes([0.25, 0, 0.65, 0.03], axisbg=axcolor)
    axli = plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=axcolor)
    axkinv = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axt0 = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

    skinv = Slider(axkinv, 'kinv', 1, 60.0, valinit=kinv0)
    st0 = Slider(axt0, 't0', 0, 10, valinit=t00)
    sli = Slider(axli, 'Li', 0, 60.0, valinit=li0)
    slf = Slider(axlf, 'Lf', 0, 60.0, valinit=lf0)

    t0 = t00
    kinv = kinv0
    li = li0
    lf = lf0

    skinv.on_changed(update)
    st0.on_changed(update)
    sli.on_changed(update)
    slf.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    bnext = Button(plt.axes([0.5, 0.9, 0.1, 0.04]), 'Save', color=axcolor, hovercolor='0.975')
    bskip = Button(plt.axes([0.61, 0.9, 0.1, 0.04]), 'Skip', color=axcolor, hovercolor='0.975')

    button.on_clicked(reset)
    bnext.on_clicked(save)
    bskip.on_clicked(skip)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()

    radio.on_clicked(colorfunc)

    set_trace(paths[0])

    plt.show()


def update(val):
    global t0, kinv, li, lf
    kinv = skinv.val
    t0 = st0.val
    li = sli.val
    lf = slf.val
    ts = np.arange(t0, 60, 0.1)
    ys = li + (lf - li) * (1 - np.exp(-(ts - t0) / kinv))
    l.set_data(ts, ys)
    fig.canvas.draw_idle()


def set_trace(path):
    global l_scatter, fig
    vs = load_csv(path, numpy=True)
    ts_dat = vs[:, 0]
    ys_dat = vs[:, 1]
    l_scatter.set_data(ts_dat - min(ts_dat), ys_dat)
    miny = min(ys_dat)
    maxy = max(ys_dat)
    rangey = maxy - miny
    ax.axis([0, 60, miny - rangey * 0.2, maxy + rangey * 0.2])
    sli.set_val(ys_dat[0])
    slf.set_val(ys_dat[-1])
    fig = plt.gcf()
    fig.canvas.set_window_title(path)


def reset(event):
    skinv.reset()
    st0.reset()
    sli.reset()
    slf.reset()


def next_trace():
    global count
    count += 1
    if count < len(paths):
        set_trace(paths[count])
    else:
        exit()


def save(event):
    global paths, count
    save_path = get_out_path(paths[count])
    ensure_folder_exists(save_path)
    save_csv(save_path, [map(str, [t0, kinv, li, lf])])
    next_trace()


def skip(event):
    next_trace()


if __name__ == "__main__":
    chdir_root()
    main('20160512-13')
