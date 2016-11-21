import luigi
from measure_kinetics_revision import TestMakeAllSlices

#
# Testing
#

def test_mk_slices():
    luigi.run(['TestMakeAllSlices', '--folder',
               '/Volumes/Mac Ext 2/Suda Electrochromism/20161013/'])
