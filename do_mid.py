"""
This file will run the two main specifications of the cstwMPC project: FBS-style
aggregate shocks, perpetual youth, matching net worth.  Will run both beta-point
and beta-dist versions.
"""

import Code.calibration as parameters
from Code.estimation import estimate
from Code.Options.all_options import all_options


basic_options = all_options["UseUniformBetaDist"].copy()
basic_options.update(all_options["DoStandardWork"])

# Run beta-point model

point_options = basic_options.copy()
point_options.update(all_options["MainSpecPoint"])

estimate(point_options, parameters)

# Run beta-dist model

dist_options = basic_options.copy()
dist_options.update(all_options["MainSpecDist"])

estimate(dist_options, parameters)
