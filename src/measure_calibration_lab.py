# -*- coding: utf-8 -*-

import os
from subprocess import call
import csv
import sys
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
from data_tools import colors10, split_trace
from scipy.optimize import curve_fit
from util import ensure_exists, basename_noext
import pandas as pd
from image_tools import get_cie_rois
from skimage import color


def read_csv(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        reader.next()
        rows = [map(float, r) for r in reader]
    vs = {}
    for r in rows:
        i = int(r[0])
        if i not in vs:
            vs[i] = []
        vs[i].append([r[1], r[2]])
    rois = {}
    for k, v in vs.iteritems():
        if len(v) % 4 != 0:
            raise ValueError('the number of points must be multiple of 4')
        rs = []
        for i in range(len(v) / 4):
            rs.append(PointROI(v[i * 4:i * 4 + 4]))
        rois[k] = rs
    return rois


chart_colors_str = """735244
c29682
627a9d
576c43
8580b1
67bdaa
d67e2c
505ba6
c15a63
5e3c6c
9dbc40
e0a32e
383d96
469449
af363c
e7c71f
bb5695
0885a1
f3f3f2
c8c8c8
a0a0a0
7a7a79
555555
343434"""


def extract_l(s):
    def h(s):
        return float(int(s, 16)) / 255

    rgb = np.zeros((1, 1, 3))
    rgb[0, 0, :] = [h(s[0:2]), h(s[2:4]), h(s[4:6])]
    lab = color.rgb2lab(rgb)
    return lab[0, 0, 0]


def sort_with_order(vs, order):
    r = np.zeros(len(order))
    for i, o in enumerate(order):
        r[i] = vs[o]
    return r


def get_image_path(movie, file):
    return '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/slices/MVI_%d/out%04d.png' % (movie, file)


def mk_cells(ps):
    # Two vectors
    x0 = ps[0][0]
    y0 = ps[0][1]
    ax = (ps[1][0] - x0) / 4
    ay = (ps[1][1] - y0) / 4
    bx = (ps[3][0] - x0) / 6
    by = (ps[3][1] - y0) / 6

    cells = []
    w = 0.1  # width ratio
    s = 4.0
    for ai in range(4):
        for bi in range(6):
            cells.append(
                RectROI(*map(int, [x0 + ax * (ai + 0.4) - s / 2 + bx * (bi + 0.4),
                                   y0 + ay * (ai + 0.3) - s / 2 + by * (bi + 0.3),
                                   s,
                                   s])))
    return cells


movie_names = [0, 7877, 7878, 7879, 7881, 7882, 7883, 7888, 7889, 7890, 7892, 7893, 7894, 7895,
               7896, 7898, 7899, 7900, 7901, 7902, 7903, 7904, 7905, 7906, 7909, 7910, 7911,
               7912, 7913, 7914, 7915]


def find_surrounding_rect(rois):
    x_min = min(map(lambda a: a[0], rois))
    x_max = max(map(lambda a: a[0] + a[2], rois))
    y_min = min(map(lambda a: a[1], rois))
    y_max = max(map(lambda a: a[1] + a[3], rois))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


class RectROI:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def values(self):
        return [self.x, self.y, self.w, self.h]

    def __repr__(self):
        return 'Rect(%d,%d,%d,%d)' % (self.x, self.y, self.w, self.h)


class PointROI:
    def __init__(self, points):
        self.points = points

    def __getitem__(self, item):
        return self.points[item]

    def rect(self):  # Return bounding rect.
        xs = map(lambda a: a[0], self.points)
        ys = map(lambda a: a[1], self.points)
        x = min(xs)
        y = min(ys)
        w = max(xs) - x
        h = max(ys) - y
        return [x, y, w, h]

    def __repr__(self):
        return 'PointROI(%s)' % self.points.__repr__()


def draw_cells(cellss, img=None, count=None):
    from itertools import chain

    cells = list(chain.from_iterable(cellss))
    for c in cells:
        r = plt.Rectangle(xy=[c.x + c.w / 2, c.y + c.h / 2], width=c.w, height=c.h, facecolor='none', edgecolor='red')
        plt.gca().add_patch(r)
    if img is None:
        plt.xlim([0, 500])
        plt.ylim([0, 500])
    else:
        plt.imshow(img)
    if count is not None:
        plt.title(str(count))
    plt.show()


def get_cie_rois_custom(path, rois):
    img = io.imread(path)
    labs = []
    img2 = np.zeros((240, 10, 3))
    count = 0
    for roi in rois:
        rgb = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        img2[count * 10:count * 10 + roi[3], 0:roi[2], :] = rgb
        lab = color.rgb2lab(rgb)
        labs.append(np.mean(lab, axis=(0, 1)))
        count += 1
    plt.imshow(img2)
    plt.show()
    return np.array(labs)


def extract_used_l(ls):
    return [ls[23], ls[22], ls[21], ls[18]]


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))

    # chart_order = [23, 12, 9, 22, 0, 14, 7, 3, 21, 17, 16, 6, 5, 10, 15, 18]
    chart_l_values = map(extract_l, chart_colors_str.split("\n"))
    chart_order = [i[0] for i in sorted(enumerate(chart_l_values), key=lambda x: x[1])]
    print(list(enumerate(chart_order)))
    plt.plot(chart_l_values, marker='o')
    plt.show()
    ordered_chart_values = sort_with_order(chart_l_values, chart_order)
    plt.plot(ordered_chart_values, marker='o')
    plt.show()

    roiss = read_csv('parameters/20161013/calibration_rois.csv')
    count = 0
    img = np.zeros((240, 1000, 3))
    line_counts = []
    lss = []
    for k, rois in roiss.iteritems():
        path = get_image_path(movie_names[int(k)], 1)
        cellss = map(mk_cells, rois)
        # print(zip(rois,cellss))
        # img2 = io.imread(path)
        # draw_cells(cellss, img2,count)
        for cells in cellss:
            ls = get_cie_rois(path, map(lambda cell: cell.values(), cells))
            # print(ls[:, 0])
            lss.append(ls[:, 0])
            if np.isnan(ls[0, 0]):
                print('NaN', movie_names[int(k)], cells)
            lab = np.zeros((240, 10, 3))
            for i in range(24):
                lab[(i * 10):(i * 10 + 10), :, 0] = ls[i, 0]
                lab[(i * 10):(i * 10 + 10), :, 1] = ls[i, 1]
                lab[(i * 10):(i * 10 + 10), :, 2] = ls[i, 2]
            rgb = color.lab2rgb(lab)
            img[:, (count * 10):(count * 10 + 10), :] = rgb
            count += 1
            # if count >= 5:
            #     break
        line_counts.append(count)
    img = img[:, 0:(count * 10), :]
    for i in line_counts:
        img[:, i * 10 - 1, :] = 1
    # plt.imshow(img)
    # plt.show()
    lss = np.array(lss).transpose()
    l_mean = np.mean(lss,axis=1)
    chart_order = [i[0] for i in sorted(enumerate(l_mean), key=lambda x: x[1])]
    lss2 = np.zeros((len(chart_order), lss.shape[1]))
    for i in range(lss.shape[1]):
        lss2[:, i] = sort_with_order(lss[:, i], chart_order)
        # lss2[:, i] /= lss2[10, i]
    plt.plot(lss)
    plt.show()
    np.savetxt('data/kinetics/20161013 calibration l values.csv',lss.transpose())
    # plt.plot(ordered_chart_values, linewidth=3)
    plt.plot(lss2)
    plt.show()


if __name__ == "__main__":
    main()
