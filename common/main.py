import os

import luigi
from cyclic_voltammetry.measure_and_plot_ox_current import PlotOxCurrent
from cyclic_voltammetry.plot_cv import PlotCV
from film_thickness.plot_thickness import PlotThickness
from final_colors.plot_final_colors import PlotFinalColors

from plot_100cycles_old import Plot100Cycles
from plot_kinetics_old import PlotOxTrace, PlotRateConstants
from plot_kinetics_space import PlotCIESpace
from plot_uvvis_lab_timecourse import PlotUVVisTimeCIELab

# FIXME: This is likely to be broken as of Feb, 2017.

class AllFigures(luigi.WrapperTask):
    def requires(self):
        tasks = [PlotCV(name='2b')
                 , PlotUVVisTimeCIELab(name1='2a', name2='3a', name3='3b')
                 , Plot100Cycles(name='3c')
                 , PlotOxTrace(name='4a')
                 , PlotFinalColors(name1='4b', name2='S4')
                 , PlotRateConstants(name1='4c', name2='S5')
                 , PlotCIESpace(name='4d')
                 , PlotThickness(name1='S1', name2='S2')
                 , PlotOxCurrent(name1='S3',name2='2b-inset')
                 ]
        for t in tasks:
            yield t


# To use luigi.cfg
os.chdir(os.path.dirname(__file__))

luigi.run(['AllFigures', '--workers', '1'])
