'''
This is the main file for the cstwMPC project.  It estimates one version of the model
each time it is executed.  The following parameters *must* be defined in the __main__
namespace in order for this file to run correctly:
    
param_name : str
    Which parameter to introduce heterogeneity in (usually DiscFac).
dist_type : str
    Which type of distribution to use (can be 'uniform' or 'lognormal').
do_param_dist : bool
    Do param-dist version if True, param-point if False.
do_lifecycle : bool
    Use lifecycle model if True, perpetual youth if False.
do_agg_shocks : bool
    Whether to solve the FBS aggregate shocks version of the model or use idiosyncratic shocks only.
do_liquid : bool
    Matches liquid assets data when True, net worth data when False.
do_tractable : bool
    Whether to use an extremely simple alternate specification of households' optimization problem.
run_estimation : bool
    Whether to actually estimate the model specified by the other options.
run_sensitivity : [bool]
    Whether to run each of eight sensitivity analyses; currently inoperative.  Order:
    rho, xi_sigma, psi_sigma, mu, urate, mortality, g, R
find_beta_vs_KY : bool
    Whether to computes K/Y ratio for a wide range of beta; should have do_param_dist = False and param_name = 'DiscFac'.
    Currently inoperative.
path_to_models : str
    Absolute path to the location of this file.
    
All of these parameters are set when running this file from one of the do_XXX.py
files in the root directory.
'''
from __future__ import division, print_function
from __future__ import absolute_import

from builtins import str
from builtins import range

import os

import numpy as np
from copy import copy, deepcopy
from time import time
from HARK.distribution import DiscreteDistribution, MeanOneLogNormal, Uniform
from HARK.utilities import get_percentiles, get_lorenz_shares, calc_subpop_avg
from HARK import Market
from Code.cstw_agents import DoWAgent, AggDoWAgent, DoWMarket, AggDoWMarket
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt

from IPython import get_ipython # Needed to test whether being run from command line or interactively

#import SetupParamsCSTW as Params

mystr = lambda number : "{:.3f}".format(number)

def process_options(options):


    # Construct the name of the specification from user options
    if options['param_name'] == 'DiscFac':
        param_text = 'beta'
    elif options['param_name'] == 'CRRA':
        param_text = 'rho'
    else:
        param_text = options['param_name']

    if options['do_lifecycle']:
        life_text = 'LC'
    else:
        life_text = 'PY'
    
    if options['do_param_dist']:
        model_text = 'Dist'
    else:
        model_text = 'Point'

    if options['do_liquid']:
        wealth_text = 'Liquid'
    else:
        wealth_text = 'NetWorth'

    if options['do_agg_shocks']:
        shock_text = 'Agg'
    else:
        shock_text = 'Ind'

    spec_name = life_text + param_text + model_text + shock_text + wealth_text

    if options['do_param_dist']:
        pref_type_count = 7       # Number of discrete beta types in beta-dist
    else:
        pref_type_count = 1       # Just one beta type in beta-point

    if options['do_agg_shocks']:
        EstimationAgentClass = AggDoWAgent
        EstimationMarketClass = AggDoWMarket
    else:
        EstimationAgentClass = DoWAgent
        EstimationMarketClass = DoWMarket

    return param_text, life_text, model_text, wealth_text, shock_text, spec_name, pref_type_count, EstimationAgentClass, EstimationMarketClass



def get_KY_ratio_difference(economy,param_name,param_count,center,spread,dist_type):
    '''
    Finds the difference between simulated and target capital to income ratio in an economy when
    a given parameter has heterogeneity according to some distribution.

    Parameters
    ----------
    economy : CstwMPCMarket
        An object representing the entire economy, containing the various AgentTypes as an attribute.
    param_name : string
        The name of the parameter of interest that varies across the population.
    param_count : int
        The number of different values the parameter of interest will take on.
    center : float
        A measure of centrality for the distribution of the parameter of interest.
    spread : float
        A measure of spread or diffusion for the distribution of the parameter of interest.
    dist_type : string
        The type of distribution to be used.  Can be "lognormal" or "uniform" (can expand).

    Returns
    -------
    diff : float
        Difference between simulated and target capital to income ratio for this economy.
    '''
    economy.assign_parameters(LorenzBool = False, ManyStatsBool = False) # Make sure we're not wasting time calculating stuff
    economy.distribute_params(param_name,param_count,center,spread,dist_type) # Distribute parameters
    economy.solve()
    diff = economy.calc_KY_ratio_difference()
    print('get_KY_ratio_difference tried center = ' + str(center) + ' and got ' + str(diff))
    return diff


def find_lorenz_distance_at_target_KY(economy,param_name,param_count,center_range,spread,dist_type):
    '''
    Finds the sum of squared distances between simulated and target Lorenz points in an economy when
    a given parameter has heterogeneity according to some distribution.  The class of distribution
    and a measure of spread are given as inputs, but the measure of centrality such that the capital
    to income ratio matches the target ratio must be found.

    Parameters
    ----------
    economy : CstwMPCMarket
        An object representing the entire economy, containing the various AgentTypes as an attribute.
    param_name : string
        The name of the parameter of interest that varies across the population.
    param_count : int
        The number of different values the parameter of interest will take on.
    center_range : [float,float]
        Bounding values for a measure of centrality for the distribution of the parameter of interest.
    spread : float
        A measure of spread or diffusion for the distribution of the parameter of interest.
    dist_type : string
        The type of distribution to be used.  Can be "lognormal" or "uniform" (can expand).

    Returns
    -------
    dist : float
        Sum of squared distances between simulated and target Lorenz points for this economy (sqrt).
    '''
    # Define the function to search for the correct value of center, then find its zero
    intermediateObjective = lambda center : get_KY_ratio_difference(economy = economy,
                                                                 param_name = param_name,
                                                                 param_count = param_count,
                                                                 center = center,
                                                                 spread = spread,
                                                                 dist_type = dist_type)
    optimal_center = brentq(intermediateObjective,center_range[0],center_range[1],xtol=10**(-6))
    economy.center_save = optimal_center

    # Get the sum of squared Lorenz distances given the correct distribution of the parameter
    economy.assign_parameters(LorenzBool = True) # Make sure we actually calculate simulated Lorenz points
    economy.distribute_params(param_name,param_count,optimal_center,spread,dist_type) # Distribute parameters
    economy.solve_agents()
    economy.make_history()
    dist = economy.calc_lorenz_distance()
    economy.assign_parameters(LorenzBool = False)
    print ('find_lorenz_distance_at_target_KY tried spread = ' + str(spread) + ' and got ' + str(dist))

    return dist

def calc_stationary_age_dstn(LivPrb,terminal_period):
    '''
    Calculates the steady state proportions of each age given survival probability sequence LivPrb.
    Assumes that agents who die are replaced by a newborn agent with t_age=0.

    Parameters
    ----------
    LivPrb : [float]
        Sequence of survival probabilities in ordinary chronological order.  Has length T_cycle.
    terminal_period : bool
        Indicator for whether a terminal period follows the last period in the cycle (with LivPrb=0).

    Returns
    -------
    AgeDstn : np.array
        Stationary distribution of age.  Stochastic vector with frequencies of each age.
    '''
    T = len(LivPrb)
    if terminal_period:
        MrkvArray = np.zeros((T+1,T+1))
        top = T
    else:
        MrkvArray = np.zeros((T,T))
        top = T-1

    for t in range(top):
        MrkvArray[t,0] = 1.0 - LivPrb[t]
        MrkvArray[t,t+1] = LivPrb[t]
    MrkvArray[t+1,0] = 1.0

    w, v = np.linalg.eig(np.transpose(MrkvArray))
    idx = (np.abs(w-1.0)).argmin()
    x = v[:,idx].astype(float)
    AgeDstn = (x/np.sum(x))
    return AgeDstn

###############################################################################
### ACTUAL WORK BEGINS BELOW THIS LINE  #######################################
###############################################################################


def main(options, Params):

    param_text, life_text, model_text, wealth_text, \
        shock_text, spec_name, pref_type_count, EstimationAgentClass, EstimationMarketClass = process_options(options)

    # Set targets for K/Y and the Lorenz curve based on the data
    if options['do_liquid']:
        lorenz_target = np.array([0.0, 0.004, 0.025,0.117])
        KY_target = 6.60
    else: # This is hacky until I can find the liquid wealth data and import it
        lorenz_target = get_lorenz_shares(Params.SCF_wealth,weights=Params.SCF_weights,percentiles=Params.percentiles_to_match)
        lorenz_long_data = np.hstack((np.array(0.0),get_lorenz_shares(Params.SCF_wealth,weights=Params.SCF_weights,percentiles=np.arange(0.01,1.0,0.01).tolist()),np.array(1.0)))
        #lorenz_target = np.array([-0.002, 0.01, 0.053,0.171])
        KY_target = 10.26

    # Set total number of simulated agents in the population
    if options['do_param_dist']:
        if options['do_agg_shocks']:
            Population = Params.pop_sim_agg_dist
        else:
            Population = Params.pop_sim_ind_dist
    else:
        if options['do_agg_shocks']:
            Population = Params.pop_sim_agg_point
        else:
            Population = Params.pop_sim_ind_point

    # Make AgentTypes for estimation
    if options['do_lifecycle']:
        DropoutType = EstimationAgentClass(**Params.init_dropout)
        DropoutType.AgeDstn = calc_stationary_age_dstn(DropoutType.LivPrb,True)
        HighschoolType = deepcopy(DropoutType)
        HighschoolType(**Params.adj_highschool)
        HighschoolType.AgeDstn = calc_stationary_age_dstn(HighschoolType.LivPrb,True)
        CollegeType = deepcopy(DropoutType)
        CollegeType(**Params.adj_college)
        CollegeType.AgeDstn = calc_stationary_age_dstn(CollegeType.LivPrb,True)
        DropoutType.update()
        HighschoolType.update()
        CollegeType.update()
        EstimationAgentList = []
        for n in range(pref_type_count):
            EstimationAgentList.append(deepcopy(DropoutType))
            EstimationAgentList.append(deepcopy(HighschoolType))
            EstimationAgentList.append(deepcopy(CollegeType))
    else:
        if options['do_agg_shocks']:
            PerpetualYouthType = EstimationAgentClass(**Params.init_agg_shocks)
        else:
            PerpetualYouthType = EstimationAgentClass(**Params.init_infinite)
        PerpetualYouthType.AgeDstn = np.array(1.0)
        EstimationAgentList = []
        for n in range(pref_type_count):
            EstimationAgentList.append(deepcopy(PerpetualYouthType))

    # Give all the AgentTypes different seeds
    for j in range(len(EstimationAgentList)):
        EstimationAgentList[j].seed = j

    # Make an economy for the consumers to live in
    market_dict = copy(Params.init_market)
    market_dict['AggShockBool'] = options['do_agg_shocks']
    market_dict['Population'] = Population
    EstimationEconomy = EstimationMarketClass(**market_dict)
    EstimationEconomy.agents = EstimationAgentList
    EstimationEconomy.KYratioTarget = KY_target
    EstimationEconomy.LorenzTarget = lorenz_target
    EstimationEconomy.LorenzData = lorenz_long_data
    if options['do_lifecycle']:
        EstimationEconomy.assign_parameters(PopGroFac = Params.PopGroFac)
        EstimationEconomy.assign_parameters(TypeWeight = Params.TypeWeight_lifecycle)
        EstimationEconomy.assign_parameters(T_retire = Params.working_T-1)
        EstimationEconomy.assign_parameters(act_T = Params.T_sim_LC)
        EstimationEconomy.assign_parameters(ignore_periods = Params.ignore_periods_LC)
    else:
        EstimationEconomy.assign_parameters(PopGroFac = 1.0)
        EstimationEconomy.assign_parameters(TypeWeight = [1.0])
        EstimationEconomy.assign_parameters(act_T = Params.T_sim_PY)
        EstimationEconomy.assign_parameters(ignore_periods = Params.ignore_periods_PY)
    if options['do_agg_shocks']:
        EstimationEconomy(**Params.aggregate_params)
        EstimationEconomy.update()
        EstimationEconomy.makeAggShkHist()

    # Estimate the model as requested
    if options['run_estimation']:
        print('Beginning an estimation with the specification name ' + spec_name + '...')

        # Choose the bounding region for the parameter search
        if options['param_name'] == 'CRRA':
            param_range = [0.2,70.0]
            spread_range = [0.00001,1.0]
        elif options['param_name'] == 'DiscFac':
            param_range = [0.95,0.995]
            spread_range = [0.006,0.008]
        else:
            print('Parameter range for ' + options['param_name'] + ' has not been defined!')

        if options['do_param_dist']:
            # Run the param-dist estimation
            paramDistObjective = lambda spread : find_lorenz_distance_at_target_KY(
                                                            economy = EstimationEconomy,
                                                            param_name = options['param_name'],
                                                            param_count = pref_type_count,
                                                            center_range = param_range,
                                                            spread = spread,
                                                            dist_type = options['dist_type'])
            t_start = time()
            spread_estimate = (minimize_scalar(paramDistObjective,bracket=spread_range,tol=1e-4,method='brent')).x
            center_estimate = EstimationEconomy.center_save
            t_end = time()
        else:
            # Run the param-point estimation only
            paramPointObjective = lambda center : get_KY_ratio_difference(
                economy = EstimationEconomy,
                param_name = options['param_name'],
                param_count = pref_type_count,
                center = center,
                spread = 0.0,
                dist_type = options['dist_type']
            )
            t_start = time()
            center_estimate = brentq(paramPointObjective,param_range[0],param_range[1],xtol=1e-6)
            spread_estimate = 0.0
            t_end = time()

        # Display statistics about the estimated model
        EstimationEconomy.assign_parameters(LorenzBool = True)
        EstimationEconomy.assign_parameters(ManyStatsBool = True)
        EstimationEconomy.distribute_params(
            options['param_name'], pref_type_count,center_estimate,spread_estimate, options['dist_type']
        )
        EstimationEconomy.solve()
        EstimationEconomy.calc_lorenz_distance()
        print('Estimate is center=' + str(center_estimate) + ', spread=' + str(spread_estimate) + ', took ' + str(t_end-t_start) + ' seconds.')
        EstimationEconomy.center_estimate = center_estimate
        EstimationEconomy.spread_estimate = spread_estimate
        EstimationEconomy.show_many_stats(spec_name)
        print('These results have been saved to ./Code/Results/' + spec_name + '.txt\n\n')


if __name__ == '__main__':
    main()