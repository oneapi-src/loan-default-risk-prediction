#!/bin/bash

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

for k in {1..10}
do

    foldername=3m
    testsize=1000000

    echo "Running inference benchmarks on model $foldername ..."
    for j in {1..4}
    do
        python run_inference.py --trained_model ../saved_models/${foldername}/intel/model_${j}.pkl --logfile ../logs/${foldername}/intel/inference.log --silent --input_file ../data/credit_risk_test.csv --size ${testsize}
        python run_inference.py --trained_model ../saved_models/${foldername}/intel/model_${j}_daal.pkl --logfile ../logs/${foldername}/intel/inference.log --is_daal_model --silent --input_file ../data/credit_risk_test.csv --size ${testsize}
    done
done
