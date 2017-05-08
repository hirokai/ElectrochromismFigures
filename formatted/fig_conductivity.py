import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import pandas as pd
from common.util import chdir_root, ensure_folder_exists
from film_thickness.plot_thickness import plot_thickness_rpm_multi


def main():
    chdir_root()
    sns.set_style('ticks')
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    plt.rcParams['lines.linewidth'] = 1
    plt.figure(figsize=(6, 4))

    df = pd.DataFrame(columns=["state", "conductivity"])
    # df.columns = ["state", "conductivity"]
    red = "Reduced"
    ox = "Oxidized"
    df = df.append({'state': red, "conductivity": 2.01}, ignore_index=True)
    df = df.append({'state': red, "conductivity": 1.45}, ignore_index=True)
    df = df.append({'state': ox, "conductivity": 11.92}, ignore_index=True)
    df = df.append({'state': ox, "conductivity": 7.90}, ignore_index=True)
    print(df)

    sns.barplot(x='state', y='conductivity', order=[red, ox], data=df, palette="Set3", capsize=0.1)
    plt.ylabel("Conductivity [S/cm]")
    plt.ylim([0, 20])
    plt.savefig(os.path.join("formatted", "dist", "Fig S11.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
