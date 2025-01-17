"""
To be run after `do_min.py`.
Depends on having files in Results.

Will test the output files in Results for similarity with the results in the original
cstwMPC paper.
"""
import pathlib

'''
os.chdir(path_to_options)
exec(open('UseUniformBetaDist.py').read())
exec(open('DoStandardWork.py').read())

# Run beta-point model
exec(open('SimpleSpecPoint.py').read())
os.chdir(path_to_models)
exec(open('cstwMPC_MAIN.py').read())

# Run beta-dist model
os.chdir(path_to_options)
exec(open('SimpleSpecDist.py').read())
os.chdir(path_to_models)
exec(open('cstwMPC_MAIN.py').read())
'''

import re


results_files_and_targers = {
    "PYbetaPointAggNetWorthResults.txt" : { ## Targets from commit 3aefa00
    ## MPC targets
    "Average MPC for all consumers" : 0.094,
    "Average MPC in the top percentile of W/Y" : 0.075,
    "Average MPC in the top decile of W/Y" : 0.075,
    "Average MPC in the top quintile of W/Y" : 0.075,
    "Average MPC in the second quintile of W/Y" : 0.075,
    "Average MPC in the middle quintile of W/Y" : 0.075,
    "Average MPC in the fourth quintile of W/Y" : 0.075,
    "Average MPC in the bottom quintile of W/Y" : 0.167,
    "Average MPC in the top percentile of y" : 0.075,
    "Average MPC in the top decile of y" : 0.076,
    "Average MPC in the top quintile of y" : 0.080,
    "Average MPC in the second quintile of y" : 0.104,
    "Average MPC in the middle quintile of y" : 0.115,
    "Average MPC in the fourth quintile of y" : 0.080,
    "Average MPC in the bottom quintile of y" : 0.090,
    "Average MPC for the employed" : 0.092,
    "Average MPC for the unemployed"  : 0.117,
    ## Quintile targets
    "bottom wealth quintile" : 82.394,
    "second wealth quintile" : 16.224, 
    "third wealth quintile" : 1.184, 
    "fourth wealth quintile" : 0.170, 
    "top wealth quintile" : 0.028
    },
    "PYbetaPointIndNetWorthResults.txt" : { ## Targets from original paper
    ## MPC targets
    "Average MPC for all consumers" : 0.099,
    "Average MPC in the top percentile of W/Y" : 0.068,
    "Average MPC in the top decile of W/Y" : 0.071,
    "Average MPC in the top quintile of W/Y" : 0.072,
    "Average MPC in the second quintile of W/Y" : 0.074,
    "Average MPC in the middle quintile of W/Y" : 0.074,
    "Average MPC in the fourth quintile of W/Y" : 0.076,
    "Average MPC in the bottom quintile of W/Y" : 0.199,
    "Average MPC in the top percentile of y" : 0.075,
    "Average MPC in the top decile of y" : 0.078,
    "Average MPC in the top quintile of y" : 0.084,
    "Average MPC in the second quintile of y" : 0.114,
    "Average MPC in the middle quintile of y" : 0.123,
    "Average MPC in the fourth quintile of y" : 0.082,
    "Average MPC in the bottom quintile of y" : 0.093,
    "Average MPC for the employed" : 0.097,
    "Average MPC for the unemployed"  : 0.131,
    ## Quintile targets
    "bottom wealth quintile" : 61.057,
    "second wealth quintile" : 28.655, 
    "third wealth quintile" : 7.285, 
    "fourth wealth quintile" : 2.207, 
    "top wealth quintile" : 0.796
    },
    "PYbetaDistAggNetWorthResults.txt" : { ## Targets from commit 3aefa00
    ## MPC targets
    "Average MPC for all consumers" : 0.216,
    "Average MPC in the top percentile of W/Y" : 0.059,
    "Average MPC in the top decile of W/Y" : 0.062,
    "Average MPC in the top quintile of W/Y" : 0.065,
    "Average MPC in the second quintile of W/Y" : 0.081,
    "Average MPC in the middle quintile of W/Y" : 0.156,
    "Average MPC in the fourth quintile of W/Y" : 0.271,
    "Average MPC in the bottom quintile of W/Y" : 0.508,
    "Average MPC in the top percentile of y" : 0.194,
    "Average MPC in the top decile of y" : 0.195,
    "Average MPC in the top quintile of y" : 0.199,
    "Average MPC in the second quintile of y" : 0.222,
    "Average MPC in the middle quintile of y" : 0.230,
    "Average MPC in the fourth quintile of y" : 0.196,
    "Average MPC in the bottom quintile of y" : 0.234,
    "Average MPC for the employed" : 0.208,
    "Average MPC for the unemployed"  : 0.330,
    ## Quintile targets
    "bottom wealth quintile" : 53.642,
    "second wealth quintile" : 33.711, 
    "third wealth quintile" : 11.306, 
    "fourth wealth quintile" : 1.319, 
    "top wealth quintile" : 0.022
    },
    "PYbetaDistIndNetWorthResults.txt" : { ## Targets from commit 3aefa00
    ## MPC targets
    "Average MPC for all consumers" : 0.263,
    "Average MPC in the top percentile of W/Y" : 0.052,
    "Average MPC in the top decile of W/Y" : 0.057,
    "Average MPC in the top quintile of W/Y" : 0.061,
    "Average MPC in the second quintile of W/Y" : 0.094,
    "Average MPC in the middle quintile of W/Y" : 0.225,
    "Average MPC in the fourth quintile of W/Y" : 0.350,
    "Average MPC in the bottom quintile of W/Y" : 0.583,
    "Average MPC in the top percentile of y" : 0.234,
    "Average MPC in the top decile of y" : 0.238,
    "Average MPC in the top quintile of y" : 0.241,
    "Average MPC in the second quintile of y" : 0.267,
    "Average MPC in the middle quintile of y" : 0.271,
    "Average MPC in the fourth quintile of y" : 0.243,
    "Average MPC in the bottom quintile of y" : 0.291,
    "Average MPC for the employed" : 0.252,
    "Average MPC for the unemployed"  : 0.406,
    ## Quintile targets
    "bottom wealth quintile" : 51.659,
    "second wealth quintile" : 33.929, 
    "third wealth quintile" : 12.248, 
    "fourth wealth quintile" : 2.134, 
    "top wealth quintile" : 0.030
    }
}

MPC_tolerance_plus = 0.01

# This is five percentage points
quintile_tolerance_plus = 7


# Regular expressions for extracting data from the results files.

for filename in results_files_and_targers:

    with open(
        pathlib.PurePath(pathlib.Path(__file__).parent,
        "../Results",
        filename)) as f:
   
        data = f.read()

        for key in results_files_and_targers[filename]:
            try:
                if 'MPC' in key:
                    MPC_re = re.compile(f"{key} is ([\d\.]+)")

                    t = results_files_and_targers[filename][key]
                    d = float(MPC_re.search(data)[1])

                    t_floor = t - MPC_tolerance_plus
                    t_ceiling = t + MPC_tolerance_plus

                    assert d > t_floor and t < t_ceiling, f"{filename}: {key} target is {t}, got {d}. Acceptable values are between {t_floor} and {t_ceiling}"

                elif 'quintile' in key:
                    quintile_re = re.compile(f"([\d\.]+)\% are in the {key}")

                    t = results_files_and_targers[filename][key]
                    d = float(quintile_re.search(data)[1])

                    t_floor = t - quintile_tolerance_plus
                    t_ceiling = t + quintile_tolerance_plus

                    ### This should be a test of being within 10% percentile
                    assert d > t_floor and d < t_ceiling, f"{filename}: {key} target is {t}, got {d}. Acceptable values are between {t_floor} and {t_ceiling}"
            except Exception as e:
                print(filename, key)
                raise e