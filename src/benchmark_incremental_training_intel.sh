#!/bin/bash

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

for k in {1..10}
do

    foldername=3m
    initialsize=3000000
    incrementalsize=1000000

    mkdir -p ../logs/${foldername}/intel
    for j in {1..4}
    do
        if [[ $j -eq 1 ]]
        then
            echo "Training initial model with $initialsize data points..."
            python run_training.py --size ${initialsize} --num_cpu 4 --train_file ../data/batches/credit_risk_train_$j.csv --test_file ../data/credit_risk_test.csv --save_model_path ../saved_models/${foldername}/intel/model_$j.pkl --intel --logfile ../logs/${foldername}/intel/performance.log >> ../logs/${foldername}/intel/fairness.log
        else
            echo "Updating model with $incrementalsize data points..."
            python run_training.py --size ${incrementalsize} --num_cpu 4 --train_file ../data/batches/credit_risk_train_$j.csv --test_file ../data/credit_risk_test.csv --trained_model ../saved_models/${foldername}/intel/model_$((j-1)).pkl --intel --save_model_path ../saved_models/${foldername}/intel/model_$j.pkl --logfile ../logs/${foldername}/intel/performance.log >> ../logs/${foldername}/intel/fairness.log
        fi
    done
done
