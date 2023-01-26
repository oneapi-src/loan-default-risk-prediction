# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914
"""
Process raw dataset for experiments
"""

import sys
sys.path.append("../../src")
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from notebooks.utils import global_utils

def load_results_dict_training():
    results_dict = []
    subfolder_names = {'stock':'xgboost 0.81','intel':'xgboost 1.4.2'}
    foldernames = {'3m':'1M'}
    for experiment_n, foldername in enumerate(foldernames.keys()):
        for increment_n in range(1,4):
            record = {}
            for subfolder_name in subfolder_names.keys():
                logfile = f"logs/{foldername}/{subfolder_name}/performance.log"
                with open(logfile, 'r') as f:
                    lines = f.readlines()
                filtered_lines = [line for line in lines if line.find('time') != -1]

                time = np.mean([float(re.findall("\d+.\d+",filtered_lines[i])[0]) for i in range(increment_n,len(filtered_lines),4)])            
                record[subfolder_names[subfolder_name]] = time
            record['Experiment'] = experiment_n + 1
            record['Update Size'] = foldernames[foldername]
            record['Incremental Round'] = f'Update {increment_n}'
            
            stock_time = record[subfolder_names['stock']]
            intel_time = record[subfolder_names['intel']]
            record['Intel speedup over stock'] = stock_time / intel_time
            results_dict.append(record)
    return results_dict

def load_results_dict_inference():
    results_dict = defaultdict(dict)
    subfolder_names = {'stock':'xgboost 0.81','intel':'xgboost 1.4.2'}
    foldernames = {'3m':'1M'}
    for experiment_n, foldername in enumerate(foldernames.keys()):
        for increment_n in range(2,5):
            results_dict['Experiment'][experiment_n * 3 + increment_n] = experiment_n + 1
            for subfolder_name in subfolder_names.keys():
                logfile = f"logs/{foldername}/{subfolder_name}/inference.log"
                with open(logfile, 'r') as f:
                    lines = f.readlines()
                    
                results_dict['Batch Size'][experiment_n * 3 + increment_n] = foldernames[foldername]
                results_dict['Round'][experiment_n * 3 + increment_n] = f'Update {increment_n}'

                if subfolder_name == 'stock':
                    filtered_lines = lines
                    time = np.mean([float(re.findall("\d+.\d+",filtered_lines[i])[0]) for i in range(increment_n,len(lines),4)])
                    results_dict[subfolder_names[subfolder_name]][experiment_n * 3 + increment_n] = time
                else:
                    filtered_lines =[line for line in lines if line.find('daal') == -1]  
                    filtered_lines_daal = [line for line in lines if line.find('daal') != -1]  
                    time = np.mean([float(re.findall("\d+.\d+",filtered_lines[i])[0]) for i in range(increment_n,len(filtered_lines),4)])
                    results_dict[subfolder_names[subfolder_name]][experiment_n * 3 + increment_n] = time
                    
                    time = np.mean([float(re.findall("\d+.\d+",filtered_lines_daal[i])[0]) for i in range(increment_n,len(filtered_lines),4)])
                    results_dict['daalpy'][experiment_n * 3 + increment_n] = float(time)
                            
            results_dict['Intel speedup over stock:1.4.2'][experiment_n * 3 + increment_n] = results_dict[subfolder_names['stock']][experiment_n * 3 + increment_n] / results_dict[subfolder_names['intel']][experiment_n * 3 + increment_n]
            results_dict['Intel speedup over stock:daal'][experiment_n * 3 + increment_n] = results_dict[subfolder_names['stock']][experiment_n * 3 + increment_n] / results_dict['daalpy'][experiment_n * 3 + increment_n]
    return results_dict

def print_inference_benchmark_table():
    df = pd.DataFrame(load_results_dict_inference())
    df = df.round(2)
    df['xgboost 0.81'] = df['xgboost 0.81'].apply(lambda x:str(x)+'s')
    df['xgboost 1.4.2'] = df['xgboost 1.4.2'].apply(lambda x:str(x)+'s')
    df['daalpy'] = df['daalpy'].apply(lambda x:str(x)+'s')
    df['% gain:1.4.2'] = df['Intel speedup over stock:1.4.2'].apply(lambda x:str(round(x-1,2))+'%')
    df['% gain:daal'] = df['Intel speedup over stock:daal'].apply(lambda x:str(round(x-1,2))+'%')
    df['Intel speedup over stock:\n1.4.2'] = df['Intel speedup over stock:1.4.2'].apply(lambda x:str(x)+'x')
    df['Intel speedup over stock:daal'] = df['Intel speedup over stock:daal'].apply(lambda x:str(x)+'x')
    return df

def print_training_benchmark_table():
    df = pd.DataFrame(load_results_dict_training())
    df = df.round(2)
    df['xgboost 0.81'] = df['xgboost 0.81'].apply(lambda x:str(x)+'s')
    df['xgboost 1.4.2'] = df['xgboost 1.4.2'].apply(lambda x:str(x)+'s')
    df['% gain'] = df['Intel speedup over stock'].apply(lambda x:str(round(x-1,2))+'&')
    df['Intel speedup over stock'] = df['Intel speedup over stock'].apply(lambda x:str(x)+'x')
    return df

def print_training_benchmark_bargraph():
    df = pd.DataFrame(load_results_dict_training())
    fig, (ax1) = plt.subplots(1,1,figsize=[14,6])
    fig.suptitle('Incremental Training Performance Gain Relative to XGBoost v0.81')
    size_list = ['1M']
    ax1.set_ylabel('Relative Performance to Stock \n (Higher is better)')
    xbg081 = df['xgboost 0.81'].iloc()[:]
    xbg142 = df['xgboost 1.4.2'].iloc()[:]

    global_utils.bar_comparison(xbg081,xbg142,'xgboost 0.81','xgboost 1.4.2',ax1,
        xlabel=f'Experiment {1} \n Incremennt Data Size = {size_list[0]}',
        xticks=['Round1','Round2','Round3'],relative=True
        )
    ax1.legend()

def print_inference_benchmark_bargraph():
    df = pd.DataFrame(load_results_dict_inference())
    fig, (ax1) = plt.subplots(1,1,figsize=[8,6])
    fig.suptitle('Incremental Inference Performance Gain Relative to XGBoost v0.81\nBatch Size = 1M\nInference using Round 3 update of Incremental Learning Model')
    width = 1
    bar_width = 0.8
    ax1.set_ylabel('Incremental Inference Performance Gain Relative to Stock \n (Higher is better)')
    i=2
    ax = ax1
    xbg081 = round(df['xgboost 0.81'].iloc()[i:]/df['xgboost 0.81'].iloc()[i:],2)
    xbg151 = round(df['xgboost 0.81'].iloc()[i:]/df['xgboost 1.4.2'].iloc()[i:],2)
    daal = round(df['xgboost 0.81'].iloc()[i:]/df['daalpy'].iloc()[i:],2)
    rects1 = ax.bar(1 - width, xbg081, bar_width, label='xgboost 0.81', color='b')
    rects2 = ax.bar(1, xbg151, bar_width, label='xgboost 1.4.2', color='deepskyblue')
    rects3 = ax.bar(1 + width, daal, bar_width, label='daalpy', color='yellow')
    ax.bar_label(rects1, labels=[str(i) + 'x' for i in xbg081], padding=3)
    ax.bar_label(rects2, labels=[str(i) + 'x' for i in xbg151], padding=3)
    ax.bar_label(rects3, labels=[str(i) + 'x' for i in daal], padding=3)
    ax.set_xlabel(f'Increment Data Size = 1M')
    ax.set_ylim([0, 3])
    ax.tick_params(
    axis='x',        
    which='both',    
    bottom=False,    
    top=False,       
    labelbottom=False)
    ax1.legend()