#!/bin/bash
# DistributionOfWealth reproduce

if ! command -v conda &> /dev/null
then
    echo "`conda` (anaconda) needs to be installed to reproduce!"
    exit
fi

conda env update -f binder/environment.yml --prefix ./condaenv/

conda_root=$(conda info -s | grep CONDA_ROOT | sed 's/CONDA_ROOT\: //')
source $conda_root/bin/activate ./condaenv

ipython do_min.py
