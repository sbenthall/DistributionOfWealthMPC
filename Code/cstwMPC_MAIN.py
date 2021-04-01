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
import HARK.ConsumptionSaving.ConsIndShockModel as ConsIndShockModel
from HARK.ConsumptionSaving.ConsAggShockModel import CobbDouglasEconomy, AggShockConsumerType
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt

from IPython import get_ipython # Needed to test whether being run from command line or interactively

import SetupParamsCSTW as Params

mystr = lambda number : "{:.3f}".format(number)

# Construct the name of the specification from user options
if param_name == 'DiscFac':
    param_text = 'beta'
elif param_name == 'CRRA':
    param_text = 'rho'
else:
    param_text = param_name
if do_lifecycle:
    life_text = 'LC'
else:
    life_text = 'PY'
if do_param_dist:
    model_text = 'Dist'
else:
    model_text = 'Point'
if do_liquid:
    wealth_text = 'Liquid'
else:
    wealth_text = 'NetWorth'
if do_agg_shocks:
    shock_text = 'Agg'
else:
    shock_text = 'Ind'
spec_name = life_text + param_text + model_text + shock_text + wealth_text

if do_param_dist:
    pref_type_count = 7       # Number of discrete beta types in beta-dist
else:
    pref_type_count = 1       # Just one beta type in beta-point

if do_agg_shocks:
    EstimationAgentClass = AggShockConsumerType
    EstimationMarketClass = CobbDouglasEconomy
else:
    EstimationAgentClass = ConsIndShockModel.IndShockConsumerType
    EstimationMarketClass = Market

class CstwMPCAgent(EstimationAgentClass):
    '''
    A slight extension of the idiosyncratic consumer type for the cstwMPC model.
    '''
    def reset(self):
        self.initialize_sim()
        self.t_age = DiscreteDistribution(
            self.AgeDstn,
            np.arange(self.AgeDstn.size),
            seed = self.RNG.randint(0,2**31-1)
        ).draw(
            self.AgentCount,
            exact_match=False
        ).astype(int)
        self.t_cycle = copy(self.t_age)
        if hasattr(self,'kGrid'):
            self.aLvlNow = self.kInit*np.ones(self.AgentCount) # Start simulation near SS
            self.aNrmNow = self.aLvlNow/self.pLvlNow

    def market_action(self):
        if hasattr(self,'kGrid'):
            self.pLvl = self.pLvlNow/np.mean(self.pLvlNow)
        self.simulate(1)

class CstwMPCMarket(EstimationMarketClass):
    '''
    A class for representing the economy in the cstwMPC model.
    '''
    reap_vars = ['aLvl','pLvl','MPCnow','TranShk','EmpNow','t_age']
    sow_vars  = [] # Nothing needs to be sent back to agents in the idiosyncratic shocks version
    const_vars = [] # ['LorenzBool','ManyStatsBool']
    track_vars = [
        'MaggNow',
        'AaggNow',
        'KtoYnow',
        'Lorenz',
        'LorenzLong',
        'MPCall',
        'MPCretired',
        'MPCemployed',
        'MPCunemployed',
        'MPCbyIncome',
        'MPCbyWealthRatio',
        'HandToMouthPct'
        ]
    dyn_vars = [] # No dynamics in the idiosyncratic shocks version

    def __init__(self,**kwds):
        '''
        Make a new instance of CstwMPCMarket.
        '''
        super().__init__(sow_vars=self.sow_vars, reap_vars=self.reap_vars,
                    const_vars=self.const_vars, track_vars=self.track_vars,
                    dyn_vars=self.dyn_vars)
        self.assign_parameters(**kwds)
        if self.AggShockBool:
            self.sow_vars=['MaggNow','AaggNow','RfreeNow','wRteNow','PermShkAggNow','TranShkAggNow','KtoLnow']
            self.dyn_vars=['AFunc']
            self.max_loops = 20

        # Save the current file's directory location for writing output:
        self.my_file_path = path_to_models


    def solve(self):
        '''
        Solves the CstwMPCMarket.
        '''
        if self.AggShockBool:
            for agent in self.agents:
                agent.getEconomyData(self)
            Market.solve(self)
        else:
            self.solve_agents()
            self.make_history()

    def reap(self):
        super().reap()

        if 'MPCnow' in self.reap_vars:
            harvest = []

            for agent in self.agents:
                harvest.append(agent.MPCnow)

            self.reap_state['MPCnow'] = harvest

        if 't_age' in self.reap_vars:
            harvest = []

            for agent in self.agents:
                harvest.append(agent.t_age)

            self.reap_state['t_age'] = harvest

        if 'EmpNow' in self.reap_vars and len(self.reap_state['EmpNow']) == 0:
            harvest = []

            for agent in self.agents:
                harvest.append(agent.EmpNow)

            self.reap_state['EmpNow'] = harvest

        for var in self.reap_vars:
            harvest = []
            shock = False

            for agent in self.agents:
                if var in agent.shocks:
                    harvest.append(agent.shocks[var])
                    shock = True

            if shock:
                self.reap_state[var] = harvest

    def mill_rule(self,aLvl,pLvl,MPCnow,TranShk,EmpNow,t_age):
        '''
        The mill_rule for this class simply calls the method calc_stats.
        '''
        self.calc_stats(
            aLvl,
            pLvl,
            MPCnow,
            TranShk,
            EmpNow,
            t_age,
            self.parameters['LorenzBool'],
            self.parameters['ManyStatsBool']
        )

        if self.AggShockBool:
            return self.calcRandW(aLvl,pLvl)
        else: # These variables are tracked but not created in no-agg-shocks specifications
            self.MaggNow = 0.0
            self.AaggNow = 0.0

    def calc_stats(
            self,
            aLvlNow,
            pLvlNow,
            MPCnow,
            TranShkNow,
            EmpNow,
            t_age,
            LorenzBool,
            ManyStatsBool
    ):
        '''
        Calculate various statistics about the current population in the economy.

        Parameters
        ----------
        aLvlNow : [np.array]
            Arrays with end-of-period assets, listed by each ConsumerType in self.agents.
        pLvlNow : [np.array]
            Arrays with permanent income levels, listed by each ConsumerType in self.agents.
        MPCnow : [np.array]
            Arrays with marginal propensity to consume, listed by each ConsumerType in self.agents.
        TranShkNow : [np.array]
            Arrays with transitory income shocks, listed by each ConsumerType in self.agents.
        EmpNow : [np.array]
            Arrays with employment states: True if employed, False otherwise.
        t_age : [np.array]
            Arrays with periods elapsed since model entry, listed by each ConsumerType in self.agents.
        LorenzBool: bool
            Indicator for whether the Lorenz target points should be calculated.  Usually False,
            only True when DiscFac has been identified for a particular nabla.
        ManyStatsBool: bool
            Indicator for whether a lot of statistics for tables should be calculated. Usually False,
            only True when parameters have been estimated and we want values for tables.

        Returns
        -------
        None
        '''
        # Combine inputs into single arrays
        aLvl = np.hstack(aLvlNow)
        pLvl = np.hstack(pLvlNow)
        age  = np.hstack(t_age)
        TranShk = np.hstack(TranShkNow)
        EmpNow = np.hstack(EmpNow)

        # Calculate the capital to income ratio in the economy
        CohortWeight = self.PopGroFac**(-age)
        CapAgg = np.sum(aLvl*CohortWeight)
        IncAgg = np.sum(pLvl*TranShk*CohortWeight)
        KtoYnow = CapAgg/IncAgg
        self.KtoYnow = KtoYnow

        # Store Lorenz data if requested
        self.LorenzLong = np.nan
        if LorenzBool:
            order = np.argsort(aLvl)
            aLvl = aLvl[order]
            CohortWeight = CohortWeight[order]
            wealth_shares = get_lorenz_shares(aLvl,weights=CohortWeight,percentiles=self.LorenzPercentiles,presorted=True)
            self.Lorenz = wealth_shares

            if ManyStatsBool:
                self.LorenzLong = get_lorenz_shares(aLvl,weights=CohortWeight,percentiles=np.arange(0.01,1.0,0.01),presorted=True)
        else:
            self.Lorenz = np.nan # Store nothing if we don't want Lorenz data

        # Calculate a whole bunch of statistics if requested
        if ManyStatsBool:
            # Reshape other inputs
            MPC  = np.hstack(MPCnow)

            # Sort other data items if aLvl and CohortWeight were sorted
            if LorenzBool:
                pLvl = pLvl[order]
                MPC  = MPC[order]
                TranShk = TranShk[order]
                age = age[order]
                EmpNow = EmpNow[order]
            aNrm = aLvl/pLvl # Normalized assets (wealth ratio)
            IncLvl = TranShk*pLvl # Labor income this period

            # Calculate overall population MPC and by subpopulations
            MPCannual = 1.0 - (1.0 - MPC)**4
            self.MPCall = np.sum(MPCannual*CohortWeight)/np.sum(CohortWeight)
            employed =  EmpNow
            unemployed = np.logical_not(employed)
            if self.T_retire > 0: # Adjust for the lifecycle model, where agents might be retired instead
                unemployed = np.logical_and(unemployed,age < self.T_retire)
                employed   = np.logical_and(employed,age < self.T_retire)
                retired    = age >= self.T_retire
            else:
                retired    = np.zeros_like(unemployed,dtype=bool)
            self.MPCunemployed = np.sum(MPCannual[unemployed]*CohortWeight[unemployed])/np.sum(CohortWeight[unemployed])
            self.MPCemployed   = np.sum(MPCannual[employed]*CohortWeight[employed])/np.sum(CohortWeight[employed])
            self.MPCretired    = np.sum(MPCannual[retired]*CohortWeight[retired])/np.sum(CohortWeight[retired])
            self.MPCbyWealthRatio = calc_subpop_avg(MPCannual,aNrm,self.cutoffs,CohortWeight)
            self.MPCbyIncome      = calc_subpop_avg(MPCannual,IncLvl,self.cutoffs,CohortWeight)

            # Calculate the wealth quintile distribution of "hand to mouth" consumers
            quintile_cuts = get_percentiles(aLvl,weights=CohortWeight,percentiles=[0.2, 0.4, 0.6, 0.8])
            wealth_quintiles = np.ones(aLvl.size,dtype=int)
            wealth_quintiles[aLvl > quintile_cuts[0]] = 2
            wealth_quintiles[aLvl > quintile_cuts[1]] = 3
            wealth_quintiles[aLvl > quintile_cuts[2]] = 4
            wealth_quintiles[aLvl > quintile_cuts[3]] = 5
            MPC_cutoff = get_percentiles(MPCannual,weights=CohortWeight,percentiles=[2.0/3.0]) # Looking at consumers with MPCs in the top 1/3
            these = MPCannual > MPC_cutoff
            in_top_third_MPC = wealth_quintiles[these]
            temp_weights = CohortWeight[these]
            hand_to_mouth_total = np.sum(temp_weights)
            hand_to_mouth_pct = []
            for q in range(1,6):
                hand_to_mouth_pct.append(np.sum(temp_weights[in_top_third_MPC == q])/hand_to_mouth_total)
            self.HandToMouthPct = np.array(hand_to_mouth_pct)

        else: # If we don't want these stats, just put empty values in history
            self.MPCall = np.nan
            self.MPCunemployed = np.nan
            self.MPCemployed = np.nan
            self.MPCretired = np.nan
            self.MPCbyWealthRatio = np.nan
            self.MPCbyIncome = np.nan
            self.HandToMouthPct = np.nan

    def distribute_params(self,param_name,param_count,center,spread,dist_type):
        '''
        Distributes heterogeneous values of one parameter to the AgentTypes in self.agents.

        Parameters
        ----------
        param_name : string
            Name of the parameter to be assigned.
        param_count : int
            Number of different values the parameter will take on.
        center : float
            A measure of centrality for the distribution of the parameter.
        spread : float
            A measure of spread or diffusion for the distribution of the parameter.
        dist_type : string
            The type of distribution to be used.  Can be "lognormal" or "uniform" (can expand).

        Returns
        -------
        None
        '''
        # Get a list of discrete values for the parameter
        if dist_type == 'uniform':
            # If uniform, center is middle of distribution, spread is distance to either edge
            param_dist = Uniform(bot=center-spread,top=center+spread).approx(N=param_count)
        elif dist_type == 'lognormal':
            # If lognormal, center is the mean and spread is the standard deviation (in log)
            tail_N = 3
            param_dist = Lognormal(mu=np.log(center)-0.5*spread**2,sigma=spread,tail_N=tail_N,tail_bound=[0.0,0.9], tail_order=np.e).approx(N=param_count-tail_N)

        # Distribute the parameters to the various types, assigning consecutive types the same
        # value if there are more types than values
        replication_factor = len(self.agents) // param_count 
            # Note: the double division is intenger division in Python 3 and 2.7, this makes it explicit
        j = 0
        b = 0
        while j < len(self.agents):
            for n in range(replication_factor):
                self.agents[j].assign_parameters(AgentCount = int(self.Population*param_dist.pmf[b]*self.TypeWeight[n]))
                exec('self.agents[j].assign_parameters(' + param_name + '= param_dist.X[b])')
                j += 1
            b += 1

    def calc_KY_ratio_difference(self):
        '''
        Returns the difference between the simulated capital to income ratio and the target ratio.
        Can only be run after solving all AgentTypes and running make_history.

        Parameters
        ----------
        None

        Returns
        -------
        diff : float
            Difference between simulated and target capital to income ratio.
        '''
        # Ignore the first X periods to allow economy to stabilize from initial conditions
        KYratioSim = np.mean(np.array(self.history['KtoYnow'])[self.ignore_periods:])
        diff = KYratioSim - self.KYratioTarget

        return diff

    def calc_lorenz_distance(self):
        '''
        Returns the sum of squared differences between simulated and target Lorenz points.

        Parameters
        ----------
        None

        Returns
        -------
        dist : float
            Sum of squared distances between simulated and target Lorenz points (sqrt)
        '''
        LorenzSim = np.mean(np.array(self.history['Lorenz'])[self.ignore_periods:],axis=0)
        dist = np.sqrt(np.sum((100*(LorenzSim - self.LorenzTarget))**2))
        self.LorenzDistance = dist

        if np.isnan(dist):
            breakpoint()

        return dist

    def show_many_stats(self,spec_name=None):
        '''
        Calculates the "many statistics" by averaging histories across simulated periods.  Displays
        the results as text and saves them to files if spec_name is not None.

        Parameters
        ----------
        spec_name : string
            A name or label for the current specification.

        Returns
        -------
        None
        '''
        # Calculate MPC overall and by subpopulations
        MPCall = np.mean(self.history['MPCall'][self.ignore_periods:])
        MPCemployed = np.mean(self.history['MPCemployed'][self.ignore_periods:])
        MPCunemployed = np.mean(self.history['MPCunemployed'][self.ignore_periods:])
        MPCretired = np.mean(self.history['MPCretired'][self.ignore_periods:])
        MPCbyIncome = np.mean(np.array(self.history['MPCbyIncome'])[self.ignore_periods:,:],axis=0)
        MPCbyWealthRatio = np.mean(np.array(self.history['MPCbyWealthRatio'])[self.ignore_periods:,:],axis=0)
        HandToMouthPct = np.mean(np.array(self.history['HandToMouthPct'])[self.ignore_periods:,:],axis=0)

        LorenzSim = np.hstack((np.array(0.0),np.mean(np.array(self.history['LorenzLong'])[self.ignore_periods:],axis=0),np.array(1.0)))
        LorenzAxis = np.arange(101,dtype=float)

        plt.plot(LorenzAxis,self.LorenzData,'-k',linewidth=1.5)
        # TODO: Fix this.
        # plt.plot(LorenzAxis,LorenzSim,'--k',linewidth=1.5)
        plt.xlabel('Income percentile',fontsize=12)
        plt.ylabel('Cumulative wealth share',fontsize=12)
        plt.ylim([-0.02,1.0])
        # if running from command line, set interactive mode on, and make figure without blocking execution
        if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
            print('Running in interactive shell (Jupyter notebook or spyder)')
            plt.show()
        else:
            print('Running in terminal; do not wait for user to close figure before moving on')
            plt.ioff()
            plt.show(block=False)
            plt.pause(2) # Give OS time to make the plot (it only draws when main thread is sleeping)

        # Make a string of results to display
        results_string = 'Estimate is center=' + str(self.center_estimate) + ', spread=' + str(self.spread_estimate) + '\n'
        results_string += 'Lorenz distance is ' + str(self.LorenzDistance) + '\n'
        results_string += 'Average MPC for all consumers is ' + mystr(MPCall) + '\n'
        results_string += 'Average MPC in the top percentile of W/Y is ' + mystr(MPCbyWealthRatio[0]) + '\n'
        results_string += 'Average MPC in the top decile of W/Y is ' + mystr(MPCbyWealthRatio[1]) + '\n'
        results_string += 'Average MPC in the top quintile of W/Y is ' + mystr(MPCbyWealthRatio[2]) + '\n'
        results_string += 'Average MPC in the second quintile of W/Y is ' + mystr(MPCbyWealthRatio[3]) + '\n'
        results_string += 'Average MPC in the middle quintile of W/Y is ' + mystr(MPCbyWealthRatio[4]) + '\n'
        results_string += 'Average MPC in the fourth quintile of W/Y is ' + mystr(MPCbyWealthRatio[5]) + '\n'
        results_string += 'Average MPC in the bottom quintile of W/Y is ' + mystr(MPCbyWealthRatio[6]) + '\n'
        results_string += 'Average MPC in the top percentile of y is ' + mystr(MPCbyIncome[0]) + '\n'
        results_string += 'Average MPC in the top decile of y is ' + mystr(MPCbyIncome[1]) + '\n'
        results_string += 'Average MPC in the top quintile of y is ' + mystr(MPCbyIncome[2]) + '\n'
        results_string += 'Average MPC in the second quintile of y is ' + mystr(MPCbyIncome[3]) + '\n'
        results_string += 'Average MPC in the middle quintile of y is ' + mystr(MPCbyIncome[4]) + '\n'
        results_string += 'Average MPC in the fourth quintile of y is ' + mystr(MPCbyIncome[5]) + '\n'
        results_string += 'Average MPC in the bottom quintile of y is ' + mystr(MPCbyIncome[6]) + '\n'
        results_string += 'Average MPC for the employed is ' + mystr(MPCemployed) + '\n'
        results_string += 'Average MPC for the unemployed is ' + mystr(MPCunemployed) + '\n'
        results_string += 'Average MPC for the retired is ' + mystr(MPCretired) + '\n'
        results_string += 'Of the population with the 1/3 highest MPCs...' + '\n'
        results_string += mystr(HandToMouthPct[0]*100) + '% are in the bottom wealth quintile,' + '\n'
        results_string += mystr(HandToMouthPct[1]*100) + '% are in the second wealth quintile,' + '\n'
        results_string += mystr(HandToMouthPct[2]*100) + '% are in the third wealth quintile,' + '\n'
        results_string += mystr(HandToMouthPct[3]*100) + '% are in the fourth wealth quintile,' + '\n'
        results_string += 'and ' + mystr(HandToMouthPct[4]*100) + '% are in the top wealth quintile.' + '\n'
        print(results_string)

        # Save results to disk
        if spec_name is not None:
            with open(self.my_file_path  + '/Results/' + spec_name + 'Results.txt','w') as f:
                f.write(results_string)
                f.close()


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

if __name__ == '__main__':

    # Set targets for K/Y and the Lorenz curve based on the data
    if do_liquid:
        lorenz_target = np.array([0.0, 0.004, 0.025,0.117])
        KY_target = 6.60
    else: # This is hacky until I can find the liquid wealth data and import it
        lorenz_target = get_lorenz_shares(Params.SCF_wealth,weights=Params.SCF_weights,percentiles=Params.percentiles_to_match)
        lorenz_long_data = np.hstack((np.array(0.0),get_lorenz_shares(Params.SCF_wealth,weights=Params.SCF_weights,percentiles=np.arange(0.01,1.0,0.01).tolist()),np.array(1.0)))
        #lorenz_target = np.array([-0.002, 0.01, 0.053,0.171])
        KY_target = 10.26

    # Set total number of simulated agents in the population
    if do_param_dist:
        if do_agg_shocks:
            Population = Params.pop_sim_agg_dist
        else:
            Population = Params.pop_sim_ind_dist
    else:
        if do_agg_shocks:
            Population = Params.pop_sim_agg_point
        else:
            Population = Params.pop_sim_ind_point

    # Make AgentTypes for estimation
    if do_lifecycle:
        DropoutType = CstwMPCAgent(**Params.init_dropout)
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
        if do_agg_shocks:
            PerpetualYouthType = CstwMPCAgent(**Params.init_agg_shocks)
        else:
            PerpetualYouthType = CstwMPCAgent(**Params.init_infinite)
        PerpetualYouthType.AgeDstn = np.array(1.0)
        EstimationAgentList = []
        for n in range(pref_type_count):
            EstimationAgentList.append(deepcopy(PerpetualYouthType))

    # Give all the AgentTypes different seeds
    for j in range(len(EstimationAgentList)):
        EstimationAgentList[j].seed = j

    # Make an economy for the consumers to live in
    market_dict = copy(Params.init_market)
    market_dict['AggShockBool'] = do_agg_shocks
    market_dict['Population'] = Population
    EstimationEconomy = CstwMPCMarket(**market_dict)
    EstimationEconomy.agents = EstimationAgentList
    EstimationEconomy.KYratioTarget = KY_target
    EstimationEconomy.LorenzTarget = lorenz_target
    EstimationEconomy.LorenzData = lorenz_long_data
    if do_lifecycle:
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
    if do_agg_shocks:
        EstimationEconomy(**Params.aggregate_params)
        EstimationEconomy.update()
        EstimationEconomy.makeAggShkHist()

    # Estimate the model as requested
    if run_estimation:
        print('Beginning an estimation with the specification name ' + spec_name + '...')

        # Choose the bounding region for the parameter search
        if param_name == 'CRRA':
            param_range = [0.2,70.0]
            spread_range = [0.00001,1.0]
        elif param_name == 'DiscFac':
            param_range = [0.95,0.995]
            spread_range = [0.006,0.008]
        else:
            print('Parameter range for ' + param_name + ' has not been defined!')

        if do_param_dist:
            # Run the param-dist estimation
            paramDistObjective = lambda spread : find_lorenz_distance_at_target_KY(
                                                            economy = EstimationEconomy,
                                                            param_name = param_name,
                                                            param_count = pref_type_count,
                                                            center_range = param_range,
                                                            spread = spread,
                                                            dist_type = dist_type)
            t_start = time()
            spread_estimate = (minimize_scalar(paramDistObjective,bracket=spread_range,tol=1e-4,method='brent')).x
            center_estimate = EstimationEconomy.center_save
            t_end = time()
        else:
            # Run the param-point estimation only
            paramPointObjective = lambda center : get_KY_ratio_difference(
                economy = EstimationEconomy,
                param_name = param_name,
                param_count = pref_type_count,
                center = center,
                spread = 0.0,
                dist_type = dist_type
            )
            t_start = time()
            center_estimate = brentq(paramPointObjective,param_range[0],param_range[1],xtol=1e-6)
            spread_estimate = 0.0
            t_end = time()

        # Display statistics about the estimated model
        EstimationEconomy.assign_parameters(LorenzBool = True)
        EstimationEconomy.assign_parameters(ManyStatsBool = True)
        EstimationEconomy.distribute_params(
            param_name, pref_type_count,center_estimate,spread_estimate, dist_type
        )
        EstimationEconomy.solve()
        EstimationEconomy.calc_lorenz_distance()
        print('Estimate is center=' + str(center_estimate) + ', spread=' + str(spread_estimate) + ', took ' + str(t_end-t_start) + ' seconds.')
        EstimationEconomy.center_estimate = center_estimate
        EstimationEconomy.spread_estimate = spread_estimate
        EstimationEconomy.show_many_stats(spec_name)
        print('These results have been saved to ./Code/Results/' + spec_name + '.txt\n\n')
