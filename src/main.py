import luigi
import os
from plot_100cycles import Plot100Cycles
from plot_kinetics import PlotOxTrace, PlotRateConstants
from plot_final_colors import PlotFinalColors
from plot_kinetics_space import PlotCIESpace
from plot_uvvis_lab_timecourse import PlotUVVisTimeCIELab
from plot_cv import PlotCV
from plot_thickness import PlotThickness
from measure_and_plot_ox_current import PlotOxCurrent


class AllFigures(luigi.WrapperTask):
    def requires(self):
        tasks = [PlotCV(name='2a')
                 , PlotUVVisTimeCIELab(name1='2b', name2='3a', name3='3b')
                 , Plot100Cycles(name='3c')
                 , PlotOxTrace(name='4a')
                 , PlotFinalColors(name1='4b', name2='S4')
                 , PlotRateConstants(name1='4c', name2='S5')
                 , PlotCIESpace(name='4d')
                 , PlotThickness(name1='S1', name2='S2')
                 , PlotOxCurrent(name='S3')
                 ]
        for t in tasks:
            yield t


# To use luigi.cfg
os.chdir(os.path.dirname(__file__))

luigi.run(['AllFigures', '--workers', '1'])
