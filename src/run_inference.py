# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run inference on a trained model
"""

import argparse
import pathlib
import logging
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from utils.preprocess import get_feature_names

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)


def main(flags):
    """Get predictions from a trained model on an input file

    Args:
        flags: training configuration
    """

    if flags.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(flags.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=flags.logfile, level=logging.DEBUG)
    logger = logging.getLogger()

    # use intel scikit learn extension
    if flags.is_daal_model:
        import daal4py as d4p

    models = joblib.load(flags.trained_model)
    preprocess = models['preprocess']
    clf = models['clf']

    # Get Predictions
    data = pd.read_csv(flags.input_file)
    if 'loan_status' in data:
        data = data.drop(['loan_status'], axis=1)
    if 'bias_variable' in data:
        data = data.drop(['bias_variable'], axis=1)

    if flags.size > 0: # for benchmarking purposes
        orig = flags.size // len(data)
        data = data.loc[np.repeat(
            data.index.values, orig + 1)]
        data = data.iloc[:flags.size]

        if orig > 1:

            # perturbing all int/float columns
            person_age = data['person_age'].values + \
                np.random.randint(-1, 1, size=len(data))
            person_income = data['person_income'].values + \
                np.random.normal(0, 10, size=len(data))
            person_emp_length = data['person_emp_length'].values + \
                np.random.randint(-1, 1, size=len(data))
            loan_amnt = data['loan_amnt'].values + \
                np.random.normal(0, 5, size=len(data))
            loan_int_rate = data['loan_int_rate'].values + \
                np.random.normal(0, 0.2, size=len(data))
            loan_percent_income = data['loan_percent_income'].values + \
                (np.random.randint(0, 100, size=len(data)) / 1000)
            cb_person_cred_hist_length = data['cb_person_cred_hist_length'].values + \
                np.random.randint(0, 2, size=len(data))

            # perturbing all binary columns
            perturb_idx = np.random.rand(len(data)) > 0.1
            random_values = np.random.choice(
                data['person_home_ownership'].unique(), len(data))
            person_home_ownership = np.where(
                perturb_idx, data['person_home_ownership'], random_values)

            perturb_idx = np.random.rand(len(data)) > 0.1
            random_values = np.random.choice(
                data['loan_intent'].unique(), len(data))
            loan_intent = np.where(
                perturb_idx, data['loan_intent'], random_values)

            perturb_idx = np.random.rand(len(data)) > 0.1
            random_values = np.random.choice(
                data['loan_grade'].unique(), len(data))
            loan_grade = np.where(
                perturb_idx, data['loan_grade'], random_values)

            perturb_idx = np.random.rand(len(data)) > 0.1
            random_values = np.random.choice(
                data['cb_person_default_on_file'].unique(), len(data))
            cb_person_default_on_file = np.where(
                perturb_idx, data['cb_person_default_on_file'], random_values)

            data = pd.DataFrame(list(zip(
                person_age, person_income, person_home_ownership,
                person_emp_length, loan_intent, loan_grade,
                loan_amnt, loan_int_rate,
                loan_percent_income, cb_person_default_on_file, 
                cb_person_cred_hist_length,
            )), columns = data.columns)

        data = data.drop_duplicates()
        assert(len(data) == flags.size)
        data.reset_index()

    fnames = get_feature_names(preprocess.named_steps['preprocessor'])
    data_out = preprocess.transform(data)
    dout = xgb.DMatrix(
        data_out,
        feature_names=fnames
    )
    if not flags.is_daal_model:
        inf_start = time.time()
        preds = clf.predict(dout)
        inf_end = time.time()
    else:
        inf_start = time.time()
        preds = d4p.gbt_classification_prediction(
            nClasses=2, resultsToEvaluate="computeClassProbabilities") \
            .compute(data_out, clf) \
            .probabilities[:, 1]
        inf_end = time.time()

    if not flags.silent:
        out = []
        for i in range(len(preds)):
            out.append({'idx': i, 'prob': preds[i]})
        print(out)
    else:
        if flags.is_daal_model:
            logger.info("Inference time (daal): %f", inf_end - inf_start)
        else:
            logger.info("Inference time : %f", inf_end - inf_start)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--is_daal_model',
        action="store_true",
        required=False,
        default=False,
        help='toggle if file is daal4py optimized'
    )

    parser.add_argument(
        '--silent',
        action="store_true",
        required=False,
        default=False,
        help='don\'t print predictions. used for benchmarking.'
    )

    parser.add_argument(
        '--size',
        type=int,
        required=False,
        default=-1,
        help='number of data entries for eval, used for benchmarking. -1 is default.'
    )

    parser.add_argument(
        '--trained_model',
        type=str,
        required=False,
        default=None,
        help="Saved trained model to incrementally update.  If None, trains a new model."
    )

    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="input file for inference"
    )

    parser.add_argument(
        '--logfile',
        type=str,
        default="",
        help="Log file to output benchmarking results to.")

    FLAGS = parser.parse_args()
    main(FLAGS)
