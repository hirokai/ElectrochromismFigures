#
# Run in common folder.
# python -m kinetics.split
#



import cPickle as pickle
from data_tools import colors10, split_trace, load_csv, save_csv
import os
from util import ensure_exists, ensure_folder_exists, basename_noext, bcolors
import csv


class SplitTraces:
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

    def set_data(self, pedot, rpm, mode, voltage, ts, vs):
        assert (pedot in [20, 30, 40, 60, 80])
        assert (rpm in [500, 1000, 2000, 3000, 4000, 5000])
        assert (mode in ['const', 'ox', 'red'])
        assert len(ts) == len(vs)
        if mode == 'ox':
            assert voltage in [0, 0.2, 0.4, 0.6, 0.8]
        self.dat[self.mk_key(pedot, rpm, mode, voltage)] = (ts, vs)

    def get_data(self, pedot, rpm, mode, voltage):
        return self.dat[self.mk_key(pedot, rpm, mode, voltage)]


def read_sample_conditions(path):
    with open(path) as f:
        reader = csv.reader(f)
        reader.next()
        sample_conditions = [{'pedot': int(row[2]), 'rpm': int(row[1])} for row in reader]
    return sample_conditions


def read_movie_conditions(path):
    with open(path) as f:
        reader = csv.reader(f)
        reader.next()
        movie_conditions = [{'name': row[0], 'mode': row[1], 'samples': [int(s) for s in row[2].split(',')]} for row in
                            reader]
    return movie_conditions


def read_all_split_traces(path):
    with open(path) as f:
        obj = pickle.load(f)
    return obj


def split_for_mode(res, ts, vs, mode, pedot, rpm):
    assert isinstance(res, SplitTraces)
    assert pedot in [20, 30, 40, 60, 80]
    assert rpm in [500, 1000, 2000, 3000, 4000, 5000]
    assert mode in ['const', 'ox', 'red']

    tss, vss = split_trace(ts, vs, range(2, 1000, 60))

    voltage_dict = {
        'const': [0.8, -0.5, 0.8, -0.5, 0.8, -0.5],
        'ox': [0, 0.2, 0.4, 0.6, 0.8],
        # Corrected on 12/7, 2016.
        # See 20160512 Suda EC amperometry/analysis/voltage_profile.jl
        'red': [0.4, 0.2, 0, -0.2, -0.5]
    }

    voltages = voltage_dict[mode]
    if mode == 'const':
        selected = zip(tss[1:], vss[1:])
    else:
        selected = zip(tss[3::2], vss[3::2])
    for i, ys in enumerate(selected):
        if i >= len(voltages):
            break
        ts = ys[0]
        vs = ys[1]
        voltage = voltages[i]
        res.set_data(pedot, rpm, mode, voltage, ts, vs)


def split_all_traces(in_csvs, movie_conditions_csv, sample_conditions_csv):
    movie_conditions = read_movie_conditions(movie_conditions_csv)
    sample_conditions = read_sample_conditions(sample_conditions_csv)
    assert len(in_csvs) == len(movie_conditions)

    res = SplitTraces()

    # Iterate through movie
    movie_num = 1
    for in_csv, movie_cond in zip(in_csvs, movie_conditions):
        vss = load_csv(in_csv, 0, numpy=True)
        assert vss.shape[1] == len(movie_cond['samples']), (in_csv, movie_cond['samples'])
        mode = movie_cond['mode']
        for sample_num, vs in zip(movie_cond['samples'], vss.T):
            cond = sample_conditions[sample_num - 1]
            pedot = cond['pedot']
            rpm = cond['rpm']
            # Triple of (mode, pedot, rpm) uniquely identifies the sample in the set of movies.
            t = range(len(vs))
            split_for_mode(res, t, vs, mode, pedot, rpm)
        movie_num += 1
    return res


def save_split_data(dat, dataset_name, p_path, folder):
    assert isinstance(dat, SplitTraces)

    # Save as separate csv files.
    for k, d in dat.dat.iteritems():
        pedot, rpm, mode, voltage = SplitTraces.get_cond_from_key(k)
        out_path = os.path.join('data', 'kinetics', folder, dataset_name,
                                '%d perc PEDOT - %d rpm' % (pedot, rpm),
                                '%s %.1f.csv' % (mode, voltage))
        ts, vs = d
        ensure_folder_exists(out_path)
        rows = map(list, zip(*[[str(t) for t in ts], [str(v) % v for v in vs]]))
        with open(out_path, 'wb') as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)

    # Also, save as a picked data.
    ensure_folder_exists(p_path)
    with open(p_path, 'wb') as f:
        pickle.dump(dat, f)


def process(name, raw=False):
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    filelist = {'20161013': ["MVI_7877.csv", "MVI_7878.csv", "MVI_7879.csv", "MVI_7881.csv", "MVI_7882.csv",
                             "MVI_7883.csv",
                             "MVI_7888.csv", "MVI_7889.csv", "MVI_7890.csv", "MVI_7892.csv", "MVI_7893.csv",
                             "MVI_7894.csv",
                             "MVI_7895.csv", "MVI_7896.csv", "MVI_7898.csv", "MVI_7899.csv", "MVI_7900.csv",
                             "MVI_7901.csv",
                             "MVI_7902.csv", "MVI_7903.csv", "MVI_7904.csv", "MVI_7905.csv", "MVI_7906.csv",
                             "MVI_7909.csv",
                             "MVI_7910.csv", "MVI_7911.csv", "MVI_7912.csv", "MVI_7913.csv", "MVI_7914.csv",
                             "MVI_7915.csv"],
                '20161019': ["MVI_7916.csv", "MVI_7917.csv", "MVI_7918.csv", "MVI_7919.csv", "MVI_7920.csv",
                             "MVI_7921.csv",
                             "MVI_7922.csv", "MVI_7923.csv", "MVI_7924.csv", "MVI_7925.csv", "MVI_7926.csv",
                             "MVI_7927.csv",
                             "MVI_7928.csv", "MVI_7929.csv", "MVI_7930.csv", "MVI_7932.csv", "MVI_7933.csv",
                             "MVI_7934.csv",
                             "MVI_7936.csv", "MVI_7937.csv", "MVI_7938.csv", "MVI_7939.csv", "MVI_7940.csv",
                             "MVI_7941.csv",
                             "MVI_7942.csv", "MVI_7943.csv", "MVI_7944.csv", "MVI_7945.csv", "MVI_7946.csv",
                             "MVI_7947.csv"]
                }
    movie_conditions_csv = os.path.join('parameters', name, 'movie_conditions.csv')
    sample_conditions_csv = os.path.join('parameters', name, 'sample_conditions.csv')
    in_csvs = filelist[name]
    split_dat = split_all_traces(
        [os.path.join('data', 'kinetics', 'raw' if raw else 'corrected', name, n) for n in in_csvs],
        movie_conditions_csv,
        sample_conditions_csv)
    assert isinstance(split_dat, SplitTraces)
    save_split_data(split_dat, name,
                    os.path.join('data', 'kinetics', 'raw_split' if raw else 'split', '%s alldata.p' % name),
                    'raw_split' if raw else 'split'
                    )


if __name__ == "__main__":
    process('20161013', raw=True)
    process('20161019', raw=True)
    process('20161013', raw=False)
    process('20161019', raw=False)
