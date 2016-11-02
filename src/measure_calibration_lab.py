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
        vs[i].append([r[3], r[2]])
    rois = {}
    for k, v in vs.iteritems():
        if len(v) % 4 != 0:
            raise ValueError('the number of points must be multiple of 4')
        rs = []
        for i in range(len(v) / 4):
            rs.append(PointROI(v[i * 4:i * 4 + 4]))
        rois[k] = rs
    return rois


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
    for ai in range(4):
        for bi in range(6):
            cells.append(
                RectROI(*map(int, [x0 + ax * ai + bx * bi + 5,
                                   y0 + ay * ai + by * bi + 5,
                                   5,
                                   5])))
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
        return [self.x,self.y,self.w,self.h]


class PointROI:
    def __init__(self, points):
        self.points = points

    def __getitem__(self, item):
        return self.points[item]

    def rect(self):
        print(self.points)
        xs = map(lambda a: a[0], self.points)
        ys = map(lambda a: a[0], self.points)
        x = min(xs)
        y = min(ys)
        w = max(xs) - x
        h = max(ys) - y
        return [x, y, w, h]


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    roiss = read_csv('parameters/20161013/calibration_rois.csv')
    count = 0
    img = np.zeros((240, 1000, 3))
    for k, rois in roiss.iteritems():
        path = get_image_path(movie_names[int(k)], 1)
        cellss = map(mk_cells, rois)
        print(cellss)
        # roi = [rois[0][0][0], rois[0][0][1], 200, 100]
        # img2 = io.imread(path)
        # rgb = img2[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        # plt.imshow(rgb)
        # plt.show()
        for cells in cellss:
            ls = get_cie_rois(path, map(lambda cell: cell.values(), cells))
            print(ls[:, 0])
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
            # print(ls)
        if count >= 5:
            break
    img = img[:, 0:(count * 10), :]
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
