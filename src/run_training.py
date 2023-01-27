# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run incremental training with bias measurements
"""

import argparse
import pathlib
import logging
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from utils.fairness import get_fairness_parity_report
from utils.preprocess import get_feature_names

plt.set_loglevel("info")
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(5)


def main(flags):
    """Benchmark incremental training with bias info

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

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PowerTransformer, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, roc_auc_score

    # load data
    data_train = pd.read_csv(flags.train_file)

    if flags.size > 0: # for benchmarking purposes
        orig = flags.size // len(data_train)
        data_train = data_train.loc[np.repeat(
            data_train.index.values, orig + 1)]
        data_train = data_train.iloc[:flags.size]

        if orig > 1:

            # perturbing all int/float columns
            person_age = data_train['person_age'].values + \
                np.random.randint(-1, 1, size=len(data_train))
            person_income = data_train['person_income'].values + \
                np.random.normal(0, 10, size=len(data_train))
            person_emp_length = data_train['person_emp_length'].values + \
                np.random.randint(-1, 1, size=len(data_train))
            loan_amnt = data_train['loan_amnt'].values + \
                np.random.normal(0, 5, size=len(data_train))
            loan_int_rate = data_train['loan_int_rate'].values + \
                np.random.normal(0, 0.2, size=len(data_train))
            loan_percent_income = data_train['loan_percent_income'].values + \
                (np.random.randint(0, 100, size=len(data_train)) / 1000)
            cb_person_cred_hist_length = data_train['cb_person_cred_hist_length'].values + \
                np.random.randint(0, 2, size=len(data_train))

            # perturbing all binary columns
            perturb_idx = np.random.rand(len(data_train)) > 0.1
            random_values = np.random.choice(
                data_train['person_home_ownership'].unique(), len(data_train))
            person_home_ownership = np.where(
                perturb_idx, data_train['person_home_ownership'], random_values)

            perturb_idx = np.random.rand(len(data_train)) > 0.1
            random_values = np.random.choice(
                data_train['loan_intent'].unique(), len(data_train))
            loan_intent = np.where(
                perturb_idx, data_train['loan_intent'], random_values)

            perturb_idx = np.random.rand(len(data_train)) > 0.1
            random_values = np.random.choice(
                data_train['loan_grade'].unique(), len(data_train))
            loan_grade = np.where(
                perturb_idx, data_train['loan_grade'], random_values)

            perturb_idx = np.random.rand(len(data_train)) > 0.1
            random_values = np.random.choice(
                data_train['cb_person_default_on_file'].unique(), len(data_train))
            cb_person_default_on_file = np.where(
                perturb_idx, data_train['cb_person_default_on_file'], random_values)

            data_train = pd.DataFrame(list(zip(
                person_age, person_income, person_home_ownership,
                person_emp_length, loan_intent, loan_grade,
                loan_amnt, loan_int_rate, data_train['loan_status'].values,
                loan_percent_income, cb_person_default_on_file, 
                cb_person_cred_hist_length, data_train['bias_variable'].values
            )), columns = data_train.columns)

        data_train = data_train.drop_duplicates()
        assert(len(data_train) == flags.size)
        data_train.reset_index()

    # Don't train on bias variable
    X_train = data_train.drop(['loan_status', 'bias_variable'], axis=1)
    y_train = data_train['loan_status']

    # define model
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'nthread': flags.num_cpu,
        'tree_method': 'hist',
        'learning_rate': 0.02,
        'max_depth': 10,
        'min_child_weight': 6,
        'n_jobs': flags.num_cpu,
        'verbosity': 0,
        'silent': 1
    }
    if flags.trained_model is None:
        num_imputer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median'))
            ]
        )
        pow_transformer = PowerTransformer()
        cat_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(transformers=[
            ('num', num_imputer, ['loan_int_rate', 'person_emp_length',
                                  'cb_person_cred_hist_length']),
            ('pow', pow_transformer, ['person_age', 'person_income',
                                      'loan_amnt', 'loan_percent_income']),
            ('cat', cat_transformer, ['person_home_ownership',
                                      'loan_intent', 'loan_grade',
                                      'cb_person_default_on_file'])
        ], remainder='passthrough')

        # separate pipeline to allow for benchmarking
        preprocess = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
    else:
        models = joblib.load(flags.trained_model)
        preprocess = models['preprocess']
        clf = models['clf']

    # Train model
    # Refit everything including the pre-processor
    X_train_out = preprocess.fit_transform(X_train)
    fnames = get_feature_names(preprocess.named_steps['preprocessor'])
    dtrain = xgb.DMatrix(
        X_train_out, y_train.values,
        feature_names=fnames
    )
    start = time.time()
    if flags.trained_model is not None:
        clf = xgb.train(params, dtrain, xgb_model=clf,
                        num_boost_round=flags.estimators)
        end = time.time()
        logger.info(
            "Incremental update training time : %f seconds", end-start)
    else:
        clf = xgb.train(params, dtrain, num_boost_round=flags.estimators)
        end = time.time()
        logger.info("New model training time : %f seconds", end-start)

    # Evaluate model performance metrics on hold out test set
    data_test = pd.read_csv(flags.test_file)
    bias_indicator = data_test['bias_variable'].values.astype(int)
    X_test = data_test.drop(['loan_status', 'bias_variable'], axis=1)
    y_test = data_test['loan_status']

    X_test_out = preprocess.transform(X_test)
    X_test_out = pd.DataFrame(
        X_test_out, columns=fnames)

    predictions = clf.predict(xgb.DMatrix(X_test_out))
    logger.info(classification_report(
        predictions > 0.5, y_test
    ))
    auc = roc_auc_score(y_test, predictions)
    logger.info("AUC : %f", auc)

    # record fairness metrics for given model on holdout test set
    parity_values = get_fairness_parity_report(
        clf, X_test_out, y_test,
        bias_indicator)

    print("Parity Ratios (Privileged/Non-Privileged):")
    for k, v in parity_values.items():
        print(f"\t{k.upper()} : {v:.2f}")

    # save model and preprocessor
    if flags.save_model_path is not None:
        path = pathlib.Path(flags.save_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'preprocess': preprocess, 'clf': clf}, path)

        if flags.intel:
            # if intel is toggled, also convert to daal4py optimized model
            import daal4py as d4p
            daal_model = d4p.get_gbt_model_from_xgboost(clf)
            fname = path.stem + "_daal" + path.suffix
            outfile = path.parent / fname
            joblib.dump({'preprocess': preprocess, 'clf': daal_model}, outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--intel',
        action="store_true",
        required=False,
        default=False,
        help='use intel daal4py for model optimization'
    )

    parser.add_argument(
        '--num_cpu',
        type=int,
        required=False,
        default=4,
        help='number of cpu cores to use'
    )

    parser.add_argument(
        '--size',
        type=int,
        required=False,
        default=-1,
        help='number of data entries to duplicate data for training and benchmarking. \
             -1 uses the original data size. Default is -1.'
    )

    parser.add_argument(
        '--trained_model',
        type=str,
        required=False,
        default=None,
        help="saved trained model to incrementally update.  If not provided, trains a new model."
    )

    parser.add_argument(
        '--save_model_path',
        type=str,
        required=False,
        default=None,
        help="path to save a trained model.  If not provided, does not save."
    )

    parser.add_argument(
        '--train_file',
        type=str,
        required=True,
        help="data file for training"
    )

    parser.add_argument(
        '--test_file',
        type=str,
        required=True,
        help="data file for testing"
    )

    parser.add_argument(
        '--logfile',
        type=str,
        default="",
        help="log file to output benchmarking results to")

    parser.add_argument(
        '--estimators',
        type=int,
        default=100,
        help=" number of estimators to use.")

    FLAGS = parser.parse_args()
    main(FLAGS)
