# -*- coding: utf-8 -*-
import csv
from skimage import io, color
import numpy as np

csv_path = '/Users/hiroyuki/Documents/Nishizawa Lab/2016 準備中の論文/20160316 Paper Suda electrochromism/20161017 測定項目/calibration_rois.csv'

files = """/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7877.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7878.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7879.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7881.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7882.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7883.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7888.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7889.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7890.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7892.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7893.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7894.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7895.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7896.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7898.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7899.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7900.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7901.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7902.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7903.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7904.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7905.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7906.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7909.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7910.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7911.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7912.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7913.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7914.MOV_out/out0060.png
/Volumes/Mac Ext 2/Suda Electrochromism/2016-10-13 Suda/MVI_7915.MOV_out/out0060.png"""


def read_rois(path):
    roiss = [[]] * 30
    with open(path) as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            if len(row) >= 3:
                roiss[int(row[0])-1].append([int(float(row[1])), int(float(row[2]))])
    return roiss


def split_into_cells(path, rois):
    rgb_whole = io.imread(path)
    ms = []
    for roi in rois:
        rgb = rgb_whole[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        lab = color.rgb2lab(rgb)
        m = np.mean(lab[:, :, 0])
        ms.append(m)
    return ms


def main():
    roiss = read_rois(csv_path)
    print(roiss)
    return
    paths = files.split('\n')
    for path, rois in zip(paths, roiss):
        split_into_cells(path, rois)


main()
