import os
from common.data_tools import load_csv


# Data for kinetics plots
class Kinetics:
    def __init__(self):
        self.dat = {}

    def __repr__(self):
        ks = self.dat.keys()
        return '%d entries: %s' % (len(ks), ' '.join(ks))

    @staticmethod
    def mk_key(pedot, rpm, mode, voltage):
        return '%d,%d,%s,%.1f' % (pedot, rpm, mode, voltage)

    @staticmethod
    def get_cond_from_key(k):
        p, r, m, v = k.split(',')
        return int(p), int(r), m, float(v)

    def set_data(self, pedot, rpm, mode, voltage, v):
        assert (pedot in [20, 30, 40, 60, 80])
        assert (rpm in [500, 1000, 2000, 3000, 4000, 5000])
        assert (mode in ['const', 'ox', 'red'])
        if mode == 'ox':
            assert voltage in [0, 0.2, 0.4, 0.6, 0.8]
        self.dat[self.mk_key(pedot, rpm, mode, voltage)] = v

    def append_data(self, pedot, rpm, mode, voltage, v):
        d = self.get_data(pedot, rpm, mode, voltage)
        if d:
            self.set_data(pedot, rpm, mode, voltage, d + [v])
        else:
            self.set_data(pedot, rpm, mode, voltage, [v])

    def get_data(self, pedot, rpm, mode, voltage):
        return self.dat.get(self.mk_key(pedot, rpm, mode, voltage))


class KineticsDataType:
    CorrectedTrace, FinalL, RateConstant, SplitTraces, RawSplitTraces = range(5)


# Todo: Stub
def read_kinetics(typ, dates):
    if typ == KineticsDataType.SplitTraces:
        dat = read_split_traces('split', dates)
    elif typ == KineticsDataType.RawSplitTraces:
        dat = read_split_traces('raw_split', dates)
    elif typ == KineticsDataType.RateConstant:
        if dates == 'predefined':
            dat = read_rate_constant(dates, alldata=True)
        else:
            dat = read_rate_constant(dates, alldata=True)
    elif typ == KineticsDataType.FinalL:
        dat = read_final_l(dates)
    else:
        raise ValueError("Unsupported operation")

    assert isinstance(dat, Kinetics)
    return dat


def read_split_traces(typ, dates=None):
    dat = Kinetics()

    if dates is None:
        dates = ['20160512-13', '20161013', '20161019']

    def read_mode(mode, voltages):
        for voltage in voltages:
            for date in dates:
                path = os.path.join('data', 'kinetics', typ, date,
                                    '%d perc PEDOT - %d rpm' % (pedot, rpm), '%s %.1f.csv' % (mode, voltage))
                if os.path.exists(path):
                    traces = load_csv(path, numpy=True)
                    dat.append_data(pedot, rpm, mode, voltage, traces)
                else:
                    print('Missing: %s' % path)

    for rpm in [500, 1000, 2000, 3000, 4000, 5000]:
        for pedot in [20, 30, 40, 60, 80]:
            read_mode('ox', [0, 0.2, 0.4, 0.6, 0.8])
            read_mode('red', [-0.5, -0.2, 0, 0.2, 0.4])

    assert isinstance(dat, Kinetics)
    return dat


def read_kinetics_predefined_dates(typ):
    dat = read_kinetics(typ, dates=['20160512-13', '20161019'])
    dat2 = read_kinetics(typ, dates=['20160512-13', '20161013'])
    voltages = [-0.5, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
    for voltage in voltages:
        rpm = 2000
        for mode in ['ox','red']:
            d = dat2.get_data(20, rpm, mode, voltage)
            if d:
                dat.set_data(20, rpm, mode, voltage, d)
    return dat


def read_final_l_predefined_dates():
    return read_kinetics_predefined_dates(KineticsDataType.FinalL)


def read_final_l(dates=None):
    dat = Kinetics()

    if dates is None:
        dates = ['20160512-13', '20161013', '20161019']

    def read_mode(mode, voltages):
        for voltage in voltages:
            for date in dates:
                path = os.path.join('data', 'kinetics', 'fitted_manual', date,
                                    '%d perc PEDOT - %d rpm' % (pedot, rpm), '%s %.1f.csv' % (mode, voltage))
                if os.path.exists(path):
                    with open(path) as f:
                        t0, kinv, li, lf = map(float, f.read().strip().split(','))
                        dat.append_data(pedot, rpm, mode, voltage, lf)

    for rpm in [500, 1000, 2000, 3000, 4000, 5000]:
        for pedot in [20, 30, 40, 60, 80]:
            read_mode('ox', [0, 0.2, 0.4, 0.6, 0.8])
            read_mode('red', [-0.5, -0.2])
    assert isinstance(dat, Kinetics)
    return dat


def read_rate_constant(dates=None, alldata=False):
    dat = Kinetics()

    if dates is None:
        dates = ['20160512-13', '20161013', '20161019']

    for rpm in [500, 1000, 2000, 3000, 4000, 5000]:
        for pedot in [20, 30, 40, 60, 80]:
            for voltage in [0.2, 0.4, 0.6, 0.8]:
                for date in dates:
                    path = os.path.join('data', 'kinetics', 'fitted_manual', date,
                                        '%d perc PEDOT - %d rpm' % (pedot, rpm), 'ox %.1f.csv' % voltage)
                    if os.path.exists(path):
                        with open(path) as f:
                            t0, kinv, li, lf = map(float, f.read().strip().split(','))
                            k = 1.0 / kinv
                            dat.append_data(pedot, rpm, 'ox', voltage, k)
            if alldata:
                for voltage in [-0.5, -0.2, 0]:
                    for date in dates:
                        path = os.path.join('data', 'kinetics', 'fitted_manual', date,
                                            '%d perc PEDOT - %d rpm' % (pedot, rpm), 'red %.1f.csv' % voltage)
                        if os.path.exists(path):
                            with open(path) as f:
                                t0, kinv, li, lf = map(float, f.read().strip().split(','))
                                k = 1.0 / kinv
                                dat.append_data(pedot, rpm, 'red', voltage, k)

    assert isinstance(dat, Kinetics)
    return dat
