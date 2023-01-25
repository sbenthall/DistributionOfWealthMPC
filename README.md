# [cstwMPC](http://econ.jhu.edu/people/ccarroll/papers/cstwMPC) Replication Made using the ARK

## To replicate estimates and output like those in the paper: 

1. Install Anaconda for Python 3
2. In the Anaconda terminal (Windows) or Unix-like terminal (other OS):
    - Navigate to ./Code/
    - run "pip install -r requirements.txt"
3. Run Spyder, and open ./do_min, ./do_mid.py, ./do_all.py, or ./do_custom.py
4. Run the code by clicking the green arrow button.

Alternatively, you can execute any of the files from the command line, e.g.

	`ipython do_min.py`

Running do_min.py will estimate two simple specifications of the model, with no aggregate shocks.

    This takes a few minutes to run-- approximately 10-15 minutes on a typical computer.

Running do_mid.py will estimate the two main specifications reported in the paper

	This takes several hours to run.

Running do_custom.py will let you choose exactly which model to estimate

Progress will be printed to screen when these files are run.

## Testing changes

This REMARK is versioned. It will be updated with new releases of HARK.

To test whether recent changes in the dependencies have significantly changed the substantive results, run the test script after running `do_min.py`:

```
python Code/Tests/test_results.py
```

This script compares the results in the output test file with target values.
It will report any results that are outside of a threshold of tolerance.
