import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from common.util import chdir_root, ensure_folder_exists
from uv_vis.compare_uvvis_cielab import plot_absorbance_l_correlation_simple, PlotColorCharts


def main():
    chdir_root()
    plt.figure(figsize=(8, 6))
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    plt.rcParams['lines.linewidth'] = 1

    # plot_absorbance_l_correlation_simple()
    # luigi.run('PlotColorCharts')
    PlotColorCharts().run()

    outpath = os.path.join('formatted', 'dist', 'Fig 3b.pdf')
    ensure_folder_exists(outpath)
    shutil.copy(str(PlotColorCharts().output()['corrected'].path), outpath)


if __name__ == "__main__":
    main()
