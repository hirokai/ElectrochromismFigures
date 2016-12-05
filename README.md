# How to use

- Run `luigid` first.
- Then, run `main.py` to execute Luigi tasks, which will generate pdf files in `dist` folder.

This still depends on some external csv files.

# Required libraries

* numpy
* scipy
* matplotlib
* Luigi

# Organization of source code

Written in Python. All source code is in `src` folder.
Some .py files have test code.

"For old data" means that it uses data for the first submission.

* calibration\_with\_colorchart.py

Input: `data/kinetics/colorchart`
Output: `data/kinetics/correction`

Use previously color chart L values to calculate brightness correction factors for PEDOT/PU films.
Color chart values and raw brightness values are calculated separately in advance.


* compare\_uvvis\_cielab.py

Correspondence between L values and absorption for a revised manuscript.

* data\_tools.py
* figure\_tools.py
* image\_tools.py
* luigi\_tools.py
* util.py

Utility functions

* kinetics\_manual\_fitting.py

Manual fitting of corrected L values to first-order reactions

* main.py

Entry point for all plotting. Not updated to the latest as of  November 29.

* make\_slices.py

Provides `mk_slices` function to make png slices from .mov files.
Since Scipy cannot open .mov files, this makes input files for later steps.

* measure\_100cycles.py

_Need to be updated to use the latest version of analysis._

100 cycle measurement of the first submission.

* measure\_and\_plot\_ox\_current.py

Plotting of oxdation current from CV.

* measure\_cie\_space.py

Spatial propagation of oxidation by L values.

* measure\_colorchart.py

Measure Lab color values of color chart with specified ROIs.

* measure\_kinetics\_revision.py

Measure rate constants with new data set.

* pedot\_voltage\_conditions.py

(For old data) Curve fitting of temporal color change.

* plot\_100cycles.py

(For old data) 100 cycles repeating.

* plot\_amperometry.py

(NOT used) Plotting amperometry data.

* plot\_cv.py

Plotting CV.

* plot\_final\_colors.py

Plotting final colors.

* plot\_kinetics.py

(For old data) Plotting kinetics.

* plot\_kinetics.py

(For new data) Plotting kinetics.

* plot\_kinetics\_space.py

(For old data) Plotting spatial propagation of kinetics.

* plot\_thickness.py

Plotting thickness of film. For both old and new data.

* plot\_uvvis\_lab\_timecourse.py

Plotting UV-Vis and L value over time as well as corrlation between them.

* tensile\_test.py

Plotting tensile testing.

* test\_calibration.py

Testing calibration with a color chart.
