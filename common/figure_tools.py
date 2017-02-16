import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
import hashlib


def ensure_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def plot_and_save(func, name, show=False):
    plt.clf()
    # print hashlib.sha256(func.func_code.co_code).hexdigest()
    res = func()
    # FIXME: Use correct relative path for all cases
    ensure_exists('dist')
    plt.savefig('dist/Fig ' + name + '.pdf')
    if show:
        plt.show()
    return res


def set_format(ax, xticks, yticks, x_minor, y_minor):
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(15)


def set_common_format():
    import seaborn as sns

    ax = plt.axes()
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(15)
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})


# D3.js category10 colors
colors10 = """#1f77b4
#ff7f0e
#2ca02c
#d62728
#9467bd
#8c564b
#e377c2
#7f7f7f
#bcbd22
#17becf""".split('\n')