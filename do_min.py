"""
This file will run the absolute minimum amount of work that actually produces
relevant output-- no aggregate shocks, perpetual youth, matching net worth.
Will run both beta-point and beta-dist versions.
"""


import Code.calibration as params
from Code.estimation import estimate
from Code.Options.all_options import all_options

basic_options = all_options["UseUniformBetaDist"].copy()
basic_options.update(all_options["DoStandardWork"])

# Run beta-point model

simple_point = basic_options.copy()
simple_point.update(all_options["SimpleSpecPoint"])

estimate(simple_point, params)

# Run beta-dist model

simple_dist = basic_options.copy()
simple_dist.update(all_options["SimpleSpecDist"])

estimate(simple_dist, params)
