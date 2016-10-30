import matplotlib.pyplot as plt
import numpy as np
import luigi
from figure_tools import plot_and_save, set_common_format
from luigi_tools import cleanup
import csv
from data_tools import colors10
import pandas as pd


def plot_thickness_pedot():
	with open('../data/plot_thickness_pedot.txt', 'r') as content_file:
		dat_str = content_file.read()

	dat = np.array(map(lambda l: map(float, l.split('\t')), dat_str.split('\n')))

	plt.figure(figsize=(4.5, 3))
	(_, caps, _) = plt.errorbar(dat[:, 0], dat[:, 1] / 1000, dat[:, 2] / 1000, lw=1, elinewidth=1)
	for cap in caps:
		cap.set_markeredgewidth(1)
	plt.scatter(dat[:, 0], dat[:, 1] / 1000, lw=0, s=10)
	plt.xlabel('PEDOT ratio [%]')
	plt.ylabel('Film thickness [um]')
	plt.axis([0, 100, 0, 6])


def plot_thickness_rpm():
	with open('../data/plot_thickness_rpm.txt', 'r') as content_file:
		dat_str = content_file.read()

	dat = np.array(map(lambda l: map(float, l.split('\t')), dat_str.split('\n')))

	w = 8
	plt.figure(figsize=(4.5, 3))
	(_, caps, _) = plt.errorbar(dat[:, 0], dat[:, 1] / 1000, dat[:, 2] / 1000, lw=1, elinewidth=1)
	for cap in caps:
		cap.set_markeredgewidth(1)
	plt.scatter(dat[:, 0], dat[:, 1] / 1000, lw=0, s=10)
	plt.xlabel('Spin coating speed [rpm]')
	plt.ylabel('Film thickness [um]')
	plt.axis([0, 6000, 0, 6])


def read_thickness():
	print('read_thickness()')
	df = pd.read_csv('../data/thickness_2016_10.csv')
	df2 = pd.DataFrame()
	df2['pedot'] = df['pedot content [wt%]']
	df2['rpm'] = df['Spin coating speed [rpm]']
	for k in ['a','b','c','d','e','f']:
		df2[k] = df[k]
	df2['mean'] = np.nanmean([df['a'],df['b'],df['c'],df['d'],df['e'],df['f']],axis=0)
	df2['std'] = np.nanstd([df['a'],df['b'],df['c'],df['d'],df['e'],df['f']],axis=0)
	print(df2.groupby(['pedot','rpm']).mean())



def plot_thickness_pedot_multi():
	with open('../data/thickness_2016_10.csv', 'r') as content_file:
		reader = csv.reader(content_file)
		reader.next()
		vs = np.array([map(float, [r[0], r[1], r[2], r[9], r[10]]) for r in reader])

	dat = {}
	for i in range(30):
		rpm = str(int(vs[i, 2]))
		if rpm not in dat:
			dat[rpm] = []
		dat[rpm].append([vs[i, 1], vs[i, 3], vs[i, 4]])
	for k, v in dat.iteritems():
		dat[k] = np.array(dat[k])
	print(dat)

	plt.figure(figsize=(4.5, 3))
	ls = []
	for i, p in enumerate([500, 1000, 2000, 3000, 4000, 5000]):
		xs = dat[str(p)][:, 0]
		ys = dat[str(p)][:, 1] / 1000
		es = dat[str(p)][:, 2] / 1000
		color = colors10[i]
		(_, caps, _) = plt.errorbar(xs, ys, es, lw=1, elinewidth=1, c=color)
		l, = plt.plot(xs, ys, lw=1, c=color, label=str(p) + " rpm")
		ls.append(l)
	plt.xlabel('PEDOT ratio [wt%]')
	plt.ylabel('Film thickness [um]')
	plt.legend(handles=ls)
	plt.axis([0, 100, 0, 8])
	plt.show()


def plot_thickness_rpm_multi():
	with open('../data/thickness_2016_10.csv', 'r') as content_file:
		reader = csv.reader(content_file)
		reader.next()
		vs = np.array([map(float, [r[0], r[1], r[2], r[9], r[10]]) for r in reader])

	dat = {}
	for i in range(30):
		pedot = str(int(vs[i, 1]))
		print(pedot)
		if pedot not in dat:
			dat[pedot] = []
		dat[pedot].append([vs[i, 2], vs[i, 3], vs[i, 4]])
	for k, v in dat.iteritems():
		dat[k] = np.array(dat[k])
	print(dat)

	plt.figure(figsize=(4.5, 3))
	ls = []
	for i, p in enumerate([20, 30, 40, 60, 80]):
		xs = dat[str(p)][:, 0]
		ys = dat[str(p)][:, 1] / 1000
		es = dat[str(p)][:, 2] / 1000
		color = colors10[i]
		(_, caps, _) = plt.errorbar(xs, ys, es, lw=1, elinewidth=1, c=color)
		l, = plt.plot(xs, ys, lw=1, c=color, label=str(p) + " wt%")
		ls.append(l)
	plt.xlabel('Spin coating speed [rpm]')
	plt.ylabel('Film thickness [um]')
	plt.legend(handles=ls)
	plt.axis([0, 6000, 0, 8])
	plt.show()


class PlotThickness(luigi.Task):
	name1 = luigi.Parameter()
	name2 = luigi.Parameter()

	def requires(self):
		return []

	def output(self):
		return [luigi.LocalTarget('../dist/Fig ' + self.name1 + '.pdf'),
		        luigi.LocalTarget('../dist/Fig ' + self.name2 + '.pdf')]

	def run(self):
		set_common_format()
		plot_and_save(plot_thickness_pedot, self.name1)
		plot_and_save(plot_thickness_rpm, self.name2)


if __name__ == "__main__":
	import os

	os.chdir(os.path.dirname(__file__))
	# plot_thickness_pedot_multi()
	# plot_thickness_rpm_multi()
	read_thickness()
	# cleanup(PlotThickness(name1='S1', name2='S2'))
	# luigi.run(['PlotThickness', '--name1', 'S1', '--name2', 'S2'])

