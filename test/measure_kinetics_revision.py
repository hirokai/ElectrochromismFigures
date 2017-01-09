import luigi
from measure_kinetics_revision import TestMakeAllSlices

#
# Testing
#

def test_mk_slices():
    luigi.run(['TestMakeAllSlices', '--folder',
               '/Volumes/ExtWork/Suda Electrochromism/20161013/'])
