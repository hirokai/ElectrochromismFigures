import luigi
import os
from plot_100cycles import Plot100Cycles
from plot_kinetics import PlotOxTrace, PlotRateConstants
from plot_final_colors import PlotFinalColors
from plot_kinetics_space import PlotCIESpace
from plot_uvvis_lab_timecourse import PlotUVVisTimeCIELab
from plot_cv import PlotCV


class AllFigures(luigi.WrapperTask):
    def requires(self):
        tasks = [PlotCV(), PlotUVVisTimeCIELab(),
                 PlotOxTrace(), PlotFinalColors(), PlotRateConstants(),
                 PlotCIESpace(), Plot100Cycles()
                 ]
        for t in tasks:
            yield t


# To use luigi.cfg
os.chdir(os.path.dirname(__file__))

luigi.run(['AllFigures', '--workers', '1'])
