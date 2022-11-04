# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914
"""
Process raw dataset for experiments
"""
import pathlib

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
np.random.seed(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_batches',
        type=int,
        required=False,
        default=1,
        help='divides data into the number of batches for incremental.'
    )

    parser.add_argument(
        '--bias_prob',
        type=float,
        required=False,
        default=0.65,
        help="probability bias_variable=0 if loan is defaulted. \
              probability bias_variable=1 if loan is not defaulted."
    )

    flags = parser.parse_args()

    data = pd.read_csv("credit_risk_dataset.csv")
    fname = 'credit_risk'

    # synthesizing biased variable
    default = data['loan_status'] == 1
    non_default = data['loan_status'] == 0

    default_bias = np.random.choice(
        [0, 1], p=[flags.bias_prob, 1-flags.bias_prob], size=len(default))
    non_default_bias = np.random.choice(
        [0, 1], p=[1-flags.bias_prob, flags.bias_prob], size=len(default))

    # bias conditional on label
    data['bias_variable'] = np.where(
        default, default_bias, non_default_bias)

    split = int(len(data) * 0.7)

    # hold out test set is always needed
    data.iloc[split:].to_csv('credit_risk_test.csv', index=False)

    train = data.iloc[:split]

    path = pathlib.Path("batches/")
    path.mkdir(parents=True, exist_ok=True)

    # Create stratified batches of the dataset to simulate incremental
    # new data coming in
    sss = StratifiedShuffleSplit(n_splits=flags.num_batches)
    for i, (_, test_idx) in enumerate(sss.split(train, train['loan_status'])):
        train.iloc[test_idx].to_csv(
            f'batches/credit_risk_train_{i + 1}.csv',
            index=False
        )
