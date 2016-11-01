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
from image_tools import get_cie_l_rois


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
        vs[i].append(r[2:])
    rois = {}
    return vs


def get_image_path(movie, file):
    return '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/slices/MVI_%d/out%04d.png' % (movie, file)


def mk_cells(ps):
    # Two vectors
    x0 = ps[0][0]
    y0 = ps[0][0]
    ax = ps[1][0] - x0
    ay = ps[1][1] - y0
    bx = ps[3][0] - x0
    by = ps[3][1] - y0

    cells = []
    for ai in range(5):
        for bi in range(4):
            cells.append(map(int, [x0 + ax * ai + bx * bi, y0 + ay * ai + by * bi, 5, 5]))
    return cells


movie_names = [0, 7877, 7878, 7879, 7881, 7882, 7883, 7888, 7889, 7890, 7892, 7893, 7894, 7895,
               7896, 7898, 7899, 7900, 7901, 7902, 7903, 7904, 7905, 7906, 7909, 7910, 7911]


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    rois = read_csv('parameters/20161013/calibration_rois.csv')
    count = 0
    for k, v in rois.iteritems():
        if len(v) == 4:
            rois = [mk_cells(v)]
        elif len(v) == 12:
            rois = [mk_cells(v[0:4]), mk_cells(v[4:8]), mk_cells(v[8:12])]
        path = get_image_path(movie_names[int(k)], 1)
        print(rois)
        for roi in rois:
            ls = get_cie_l_rois(path, roi)
            print(ls)
        count += 1
        if count >= 5:
            break


if __name__ == "__main__":
    main()
