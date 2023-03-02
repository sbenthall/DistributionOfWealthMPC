"""
This file will run the absolute minimum amount of work that actually produces
relevant output-- no aggregate shocks, perpetual youth, matching net worth.
Will run both beta-point and beta-dist versions.
"""


import Code.SetupParamsCSTW as Params
from Code.cstwMPC_MAIN import estimate  # TODO better name for this
from Code.Options.all_options import all_options

basic_options = all_options["UseUniformBetaDist"].copy()
basic_options.update(all_options["DoStandardWork"])


options = basic_options.copy()
options.update(all_options["SimpleSpecPoint"])

estimate(options, Params)

options = basic_options.copy()

options.update(all_options["SimpleSpecDist"])

estimate(options, Params)
